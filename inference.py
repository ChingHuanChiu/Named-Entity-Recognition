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
from src.train.metric.ner_metric import NerMetric
from src.train.metric.counter import BIONerCounter 

from src.extract.extract import BIOExtracter
from src.data.preprocessor import IPreprocessor, NerPreprocessor
from src.model.ner import NERBertBiLSTMWithCRF, NERBertWithCRF
from src.model.util import load_model_from_checkpoint



CKPT_PATH = './storage/ckpt/bilstm_bert_crf/ckpt_ep_50_bs_8model_epoch3.pkl'
DEVICE = 'cuda:0'

MAP_DICT = {2: "B_Thing", 4: "B_Person", 6: "B_Location", 8: "B_Time", 10: 'B_Metric', 12: "B_Organization", 14: "B_Abstract", 16: "B_Physical", 18: "B_Term", 3: "I_Thing", 5: "I_Person", 7: "I_Location", 9: "I_Time", 11: "I_Metric", 13: "I_Organization", 15: "I_Abstract", 17: "I_Physical", 19: "I_Term", 1: "O", 0: "[PAD]", 20: "[SEP]"}

TOKENIZER = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)




def get_prediction(data: str, model: torch.nn.Module, preprocessor) -> pd.DataFrame:
    context = []
    # decode_seq = []
    
    with torch.no_grad():
        
            
        trans_x = preprocessor.transform(data)
        trans_x = {k:v.squeeze().view(1, -1).to(DEVICE) for k, v in trans_x.items()}

        

        output: List[List[int]] = model(trans_x) # decode sequence
        map_ = lambda t: MAP_DICT[t]
        
        
        tag = np.vectorize(map_)(output[0]).tolist()
            # decode_seq.append(o)
        
    return tag[: len(data)]





if __name__ == '__main__':
    from pprint import pprint

   
    tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)
    ner_preprocessor = NerPreprocessor(tokenizer=tokenizer)


    model = NERBertBiLSTMWithCRF(21, False, 1, local_rank=0)
    model = load_model_from_checkpoint(device=DEVICE, model=model, model_ckpt=CKPT_PATH, is_ddp_model=True)
    model.to(DEVICE)
    
    extracter = BIOExtracter()
    while True:
        input_context = input('輸入一段文字:')
        if input_context == '掰掰':
            print('掰掰 !')
            break
        tag = get_prediction(input_context, model, ner_preprocessor)
        res = {}
        print('修但幾咧.....')
        for entity_type in ['Thing', 'Person', 'Location', 'Time', 'Metric', 'Organization', 'Abstract', 'Physical', 'Term']:
            res[entity_type] = extracter.extract(content=input_context, tag=tag, entity_type=entity_type)
        pprint(res)
        print('=' * 50)




