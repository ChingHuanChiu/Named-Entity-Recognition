import torch
from fastapi import APIRouter
from fastapi import Request
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer



from src.extract.extract import BIOExtracter, IExtract
from src.model.config import HUGGINGFACE_MODEL
from src.model.ner import NERBertBiLSTMWithCRF
from src.model.util import load_model_from_checkpoint
from app.services.inference import get_inference


router = APIRouter(tags=['NER'])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_PATH = "./storage/ckpt/ner.pkl"


MODEL =  NERBertBiLSTMWithCRF(num_label=19, 
                              lstm_num_layers=1, 
                              local_rank=0,
                              device=DEVICE)

MODEL = load_model_from_checkpoint(device=DEVICE, 
                                   model=MODEL, 
                                   model_ckpt=CKPT_PATH, 
                                   is_ddp_model=True)

MODEL.to(DEVICE)
MODEL.eval()


def initial_setting():
    

    tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)

    extractor = BIOExtracter(MODEL, DEVICE, tokenizer)
    
    return extractor


templates = Jinja2Templates(directory="app/views/")

@router.get('/')
async def form_post(request: Request):
    
    return templates.TemplateResponse('index.html',
                                      context={'request': request,
                                               'result': ''
                                               }
                                      )

    

EXTRACTOR = None
@router.post('/')
async def form_post(request: Request):

    post_data = await request.form()
    
    context = post_data['context']
    global EXTRACTOR
    if EXTRACTOR is None:
        EXTRACTOR = initial_setting()
        
    
    result = get_inference(EXTRACTOR, context)

    
    return templates.TemplateResponse('index.html',
                                       context={'request': request,
                                                'result': result
                                               }
                                     )
    
    
    
    
    
        
    
        
        
    

