

import pandas as pd
from transformers import AutoTokenizer


from src.model.config import HUGGINGFACE_MODEL

from src.extract.extract import BIOExtracter, IExtract
from src.model.ner import NERBertBiLSTMWithCRF, NERBertWithCRF
from src.model.util import load_model_from_checkpoint


def get_prediction(context: str, extracter: IExtract) -> pd.DataFrame:
   
    result_dict = extracter.extract(context)
        
    return result_dict




if __name__ == '__main__':
    from pprint import pprint
    CKPT_PATH = './storage/ckpt/ckpt_ep_20_bs_8model_epoch16.pkl'
    DEVICE = 'cpu'#'cuda:0'
    TOKENIZER = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)

    model = NERBertBiLSTMWithCRF(19, 1, local_rank=0, device=DEVICE)
    model = load_model_from_checkpoint(device=DEVICE, model=model, model_ckpt=CKPT_PATH, is_ddp_model=True)
    model.to(DEVICE)
    model.eval()
    
    extractor = BIOExtracter(model=model, device=DEVICE, tokenizer=TOKENIZER)
    while True:
        input_context = input('輸入一段文字:')
        if input_context == '掰掰':
            print('掰掰 !')
            break
        res = get_prediction(input_context, extractor)
        pprint(res)
        print('=' * 50)




