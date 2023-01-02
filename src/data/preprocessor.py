from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Optional, Dict, Union

import numpy as np
import torch.nn.functional as F
import torch

from src.model.config import SEQ_MAX_LENGTH
from src.model.config import SpecialToken



class IPreprocessor(metaclass=ABCMeta):

    @abstractmethod
    def transform(sefl, *args, **kwargs):
        return NotImplemented("not implemented")

    
    
class NerPreprocessor(IPreprocessor):

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer


    def transform(self, x: str, y: Optional[List[int]] = None) -> Union[Tuple, Dict]:
        
        x = self.tokenizer([x], 
                    padding='max_length', 
                    truncation=True, 
                    max_length= SEQ_MAX_LENGTH, 
                    return_tensors='pt')
        
        if y is not None:
            assert isinstance(y, list), f'y must be type of list, {type(y)} insted'
            # append [SEP] tokenize id, 20, for transformers tokenizer
            y.append(20)
            y = torch.Tensor(y)
            # label padding do not contain [CLS] token 
            y = F.pad(y, (0, SEQ_MAX_LENGTH - y.size()[0] - 1), mode='constant', value=0)


            return x, y
        return x