import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import numpy as np
import pandas as pd
from functools import partial

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from ast import literal_eval
from transformers import BertModel


from src.model.config import HUGGINGFACE_MODEL, SEQ_MAX_LENGTH
from src.model.util import load_model_from_checkpoint
from src.train.tensorboard import TensorBoard 
from src.train.metric.ner_metric import BIONerMetric


from src.train.abstract_class.trainer import DDPTrainer
from src.model.ner import NERBertBiLSTMWithCRF, NERBertCNNWithCRF, NERBertWithCRF

tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)



class NerDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.df = dataframe
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        data  = self.df.iloc[idx, :]
        x, y = data['context'], data['tag_id']

        
        return x, y
    
    
def adjust_label_collote_fn(batch_examples, tokenizer):
    """adjust the label, the following example:
    content : 德國 metz 美茲 52af-1 digital 閃光燈
    tokenize : '[CLS]','德', '國', 'me', '##tz', '美', '茲', '52', '##af', '-', '1', 'digital', '閃', '光', '燈', '[SEP]', '[PAD]'
    origin label : [0, 0, 0, 1, 2, 2, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 0, 0]
    adjust label : [0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 3, 4, 4, 0, 0]
    """
    batch_adjust_tags = []
    
    batch_content = [i[0] for i in batch_examples]
    batch_tokenize_content = tokenizer(batch_content, 
                                        padding='max_length', 
                                        truncation=True, 
                                        max_length= SEQ_MAX_LENGTH, 
                                        return_tensors='pt',
                                        return_offsets_mapping=True)
    
    for example, offset in zip(batch_examples, batch_tokenize_content.pop('offset_mapping').squeeze(0)):
        adjust_tags = []
        example_y = example[1]
        for start, end in offset:

            if not start == end : 
                if end - start != 1:
                    # label is decided by fisrt token which is from tokenizer offset mapping
                    adjust_tags += [example_y[start]]
                else:
                    adjust_tags += example_y[start:end]

            else:
                adjust_tags += [0]
        batch_adjust_tags.append(adjust_tags)
    
    return batch_tokenize_content, torch.tensor(batch_adjust_tags)
        
        

    
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
                                                               find_unused_parameters=True
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
        

        
        decode_sequence = [F.pad(torch.Tensor(i), (0, SEQ_MAX_LENGTH - len(i)), mode='constant', value=0) for i in decode_sequence]
        
        decode_sequence = torch.stack(decode_sequence)
        
        
        loss = -1 * crf_log_likelihood
        
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
            val_writer = SummaryWriter(f'/gcs/pchome-hadoopincloud-hadoop/user/stevenchiou/pytorch/ner/v2.0.1.4/tensorboard/val/tb_ep_{self.epochs}_bs_{self.bs}')
            # val_writer = SummaryWriter('./')

        val_metric = self.metric
        self.model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for X_val, y_val in self.val_dataloader:
                X_val = {k : v.to(self.device) for k, v in X_val.items()}
                y_val = y_val.long().to(self.device)
                val_crf_log_likelihood, decode_sequence = self.model.forward(X_val, y_val)
                
                decode_sequence = [F.pad(torch.Tensor(i), (0, SEQ_MAX_LENGTH - len(i)), mode='constant', value=0) for i in decode_sequence]
                decode_sequence = torch.stack(decode_sequence)
                val_loss = -1 * val_crf_log_likelihood 
                val_running_loss += val_loss.item()
                

                y_true_val, y_pred_val = self.gather_all(y_val.to(dist.get_rank())), self.gather_all(decode_sequence.to(dist.get_rank()))
                y_true_val = torch.cat(y_true_val)
                
                y_pred_val = torch.cat(y_pred_val)
     
                
                if RANK == 0:                    
                    
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
    parser.add_argument('--epochs', default=20, type=int, required=False, help='Epochs')
    parser.add_argument('--bs', default=8, type=int, required=False, help='batch size')
    parser.add_argument('--lr', default=3e-5, type=float, required=False, help='learning rate')
    parser.add_argument('--warmup_steps', default=2480, type=int, required=False, help='warm up步數')
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

    

    train_dataset = NerDataset(dataframe=train)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BS, sampler=train_sampler, collate_fn=partial(adjust_label_collote_fn, tokenizer=tokenizer))


    val_dataset = NerDataset(dataframe=val)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=BS, sampler=val_sampler, collate_fn=partial(adjust_label_collote_fn, tokenizer=tokenizer))

    
    NUM_TRAINING_STEPS = EPOCHS * len(train_loader)
    model = NERBertBiLSTMWithCRF(num_label=19, lstm_num_layers=1, local_rank=local_rank)#NERBertWithCRF(5)
    model.init_weights()
    
    
    ner_trainer = NerTrainer(model=model, 
                             metric= BIONerMetric(), 
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
                    # checkpoint_path='./',
                    # tensorboard_path='.',
                    checkpoint_path=f'/gcs/pchome-hadoopincloud-hadoop/user/stevenchiou/pytorch/ner/v2.0.1.4/ckpt_ep_{EPOCHS}_bs_{BS}',
                    tensorboard_path=f'/gcs/pchome-hadoopincloud-hadoop/user/stevenchiou/pytorch/ner/v2.0.1.4/tensorboard/train/tb_ep_{EPOCHS}_bs_{BS}',
                    local_rank=local_rank,
                    sampler=train_sampler,
                    earlystopping_tolerance=10
                        )
    
if __name__ == '__main__':

    main()