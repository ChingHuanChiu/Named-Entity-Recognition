import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from ast import literal_eval




from src.model.config import HUGGINGFACE_MODEL, SEQ_MAX_LENGTH

from src.train.tensorboard import TensorBoard 
from src.train.metric.ner_metric import NerMetric
from src.train.metric.counter import BIONerCounter 



from src.data.preprocessor import IPreprocessor, NerPreprocessor
from src.train.abstract_class.trainer import DDPTrainer
from src.model.ner import NERBertBiLSTMWithCRF, NERBertWithCRF

tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)


class NerDataset(Dataset):
    def __init__(self, training: bool, dataframe: pd.DataFrame, preprocessor: IPreprocessor = None):
        super().__init__()
        self.training = training
        self.preprocessor = preprocessor
        self.df = dataframe
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        data  = self.df.iloc[idx, :]
        x, y = data['context'], data['tag_id']
        x, y = self.preprocessor.transform(x, y)
        x = {k:v.squeeze() for k, v in x.items()}
        
        if self.training:
            return x, y
        
        
        return x

    
class NerTrainer(DDPTrainer):
    def __init__(self, model, metric, val_dataloader, initial_lr, warm_up_step, num_training_steps, local_rank, epochs, bs):
        self.device = torch.device("cuda", local_rank)
        print(f'using device: {self.device}')
        super().__init__()
        self.model = model.to(self.device)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = torch.nn.parallel.DistributedDataParallel(
                                                               self.model, 
                                                               device_ids=[local_rank],
                                                               output_device=local_rank,
                                                               find_unused_parameters=True# cause I remove the [CLS] token
                                                                )
        
        
        self.metric = metric
        self.val_dataloader = val_dataloader
        
        # crf and lstm lr must be larger than bert
        bert_layer = list(map(id, self.model.module.bert.parameters()))
        other_layer = filter(lambda p: id(p) not in bert_layer, self.model.module.parameters())
        self.optimizer = torch.optim.Adam([{'params': self.model.module.bert.parameters(), 'lr': initial_lr},
                                           {'params': other_layer, 'lr': initial_lr * 100},
                                          ], lr=0.001)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
                                                            optimizer=self.optimizer,
                                                            num_warmup_steps=warm_up_step,
                                                            num_training_steps=num_training_steps
                                                             )
        
        self.epochs = epochs
        self.bs = bs

    def train_step(self, X_batch, y_batch):
        X_batch = {k : v.to(self.device) for k, v in X_batch.items()}
        y_batch = y_batch.long().to(self.device)
        crf_log_likelihood, decode_sequence = self.model.forward(X_batch, y_batch)
        
        decode_sequence = [F.pad(torch.Tensor(i), (0, SEQ_MAX_LENGTH - len(i) - 1), mode='constant', value=0) for i in decode_sequence]
        decode_sequence = torch.stack(decode_sequence)
        
        loss = -1 * crf_log_likelihood
        # loss = loss / self.bs
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

        self.optimizer.step()
        
        self.lr_scheduler.step()
        
        return loss, decode_sequence
    
        
    def validation_loop(self, epoch):
        VALSTEP_PER_EPOCH = len(self.val_dataloader)
        RANK = dist.get_rank()
        
        if RANK == 0:
            val_writer = SummaryWriter(f'./storage/tensorboard/val/tb_ep_{self.epochs}_bs_{self.bs}')
        

        val_metric = self.metric
        self.model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for X_val, y_val in self.val_dataloader:
                X_val = {k : v.to(self.device) for k, v in X_val.items()}
                y_val = y_val.long().to(self.device)
                val_crf_log_likelihood, decode_sequence = self.model.forward(X_val, y_val)
                
                decode_sequence = [F.pad(torch.Tensor(i), (0, SEQ_MAX_LENGTH - len(i) - 1), mode='constant', value=0) for i in decode_sequence]
                decode_sequence = torch.stack(decode_sequence)
                val_loss = -val_crf_log_likelihood #/ self.bs
                val_running_loss += val_loss.item()
                
                y_true_val, y_pred_val = self.gather_all(y_val.to(dist.get_rank())), self.gather_all(decode_sequence.to(dist.get_rank()))
                y_true_val = torch.cat(y_true_val)
                y_pred_val = torch.cat(y_pred_val)
                if RANK == 0:
                    # val_writer = SummaryWriter('./storage/tensorboard/val')
                    

                    
                    val_metric.calculate_metric(y_true_val.cpu(), y_pred_val.cpu())
                dist.barrier()
                    
                    
            rank_epoch_val_loss = val_running_loss / VALSTEP_PER_EPOCH
            sum_rank_val_epoch_loss = self.reduce_sum_all(rank_epoch_val_loss)
            epoch_val_loss = sum_rank_val_epoch_loss.cpu().numpy() / self.world_size

            if RANK == 0:
                
                print("="*30)
                print(f"Validation Loss : {epoch_val_loss}")
                ValTB = TensorBoard(val_writer)

                for name, result in val_metric.get_result().items():
                    if isinstance(result, (float, int)):
                        print(f'Validation {name} over epoch : {float(result)}')
                print('='*30)
                ValTB.start_to_write(metrics_result=val_metric.get_result(),
                                     loss=epoch_val_loss,
                                     step=epoch)
                val_metric.reset()
            dist.barrier()
        return epoch_val_loss
    


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--epochs', default=50, type=int, required=False, help='Epochs')
    parser.add_argument('--bs', default=8, type=int, required=False, help='batch size')
    parser.add_argument('--lr', default=3e-5, type=float, required=False, help='learning rate')
    parser.add_argument('--warmup_steps', default=100, type=int, required=False, help='warm up步數')
    args = parser.parse_args()
    
    dist.init_process_group(backend='nccl')
    dist.barrier()
    
    initial_lr = args.lr
    warmup_step = args.warmup_steps
    BS = args.bs
    EPOCHS=args.epochs
    
    local_rank = int(os.environ["LOCAL_RANK"])
    
    train = pd.read_csv('./storage/data/train.csv', converters={'tag_id': literal_eval})
    val = pd.read_csv('./storage/data/validation.csv', converters={'tag_id': literal_eval})
    
    
    ner_preprocessor = NerPreprocessor(tokenizer=tokenizer)

    train_dataset = NerDataset(training=True, dataframe=train, preprocessor=ner_preprocessor)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BS, sampler=train_sampler)


    val_dataset = NerDataset(training=True, dataframe=val, preprocessor=ner_preprocessor)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=BS, sampler=val_sampler)

    print(f'rank_{dist.get_rank()}_train_loader', len(train_loader))
    NUM_TRAINING_STEPS = EPOCHS * len(train_loader)
    model = NERBertBiLSTMWithCRF(21, False, 1, local_rank=local_rank)#NERBertWithCRF(21, False)#
    model.init_weights()
    ner_trainer = NerTrainer(model=model, 
                           metric= NerMetric(counter=BIONerCounter), 
                           val_dataloader=val_loader, 
                           initial_lr=initial_lr, 
                           warm_up_step=warmup_step, 
                           num_training_steps=NUM_TRAINING_STEPS, 
                           local_rank=local_rank,
                           epochs = EPOCHS,
                           bs=BS
                          )
    
    
    ner_trainer.start_to_train(
                    train_data_loader=train_loader,
                    epochs=EPOCHS,
                    checkpoint_path=f'./storage/ckpt_ep_{EPOCHS}_bs_{BS}/',
                    tensorboard_path=f'./storage/tensorboard/train/tb_ep_{EPOCHS}_bs_{BS}',
                    local_rank=local_rank,
                    sampler=train_sampler,
                    earlystopping_tolerance=10
                        )
    
if __name__ == '__main__':

    main()