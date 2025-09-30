from typing import List, Any, Dict, Tuple
from abc import ABCMeta, abstractmethod
from collections import defaultdict


import numpy as np
import pandas as pd
import torch

from src.model.config import SEQ_MAX_LENGTH, labelTOtags


class IExtract(metaclass=ABCMeta):

    @abstractmethod
    def extract(self, *arg, **kwargs):
        raise NotImplemented("not implemented !")


class BIOExtracter(IExtract):
    
    def __init__(self, model: torch.nn.Module, device: str, tokenizer) -> None:
        
        self.model = model
        self.device = device
        self.tokenizer = tokenizer

    def extract(self, content: str) -> Dict[str, str]:
        results = []
        with torch.no_grad():
            inputs = self.tokenizer(content, truncation=True, return_tensors="pt", return_offsets_mapping=True)
            # remove [CLS] and [SEP] 
            offsets = inputs.pop('offset_mapping').squeeze(0)[1: -1]
            inputs = inputs.to(self.device)
            
            predictions = self.model(inputs)[0][1: -1]
    
            pred_label = defaultdict(str)
            idx = 0
            while idx < len(predictions):
                pred = predictions[idx]
                label = labelTOtags[pred]
                
                word_ids = inputs.word_ids()[1: -1]
                idx_at_word_ids_list = self.find_indices(word_ids, word_ids[idx])
                
            
                if label != "O":
                    
                    label = label[2:] # Remove the B- or I-
                    start, end = offsets[idx]
                    
                    if len(idx_at_word_ids_list) > 1:
                        start, end = self._adjust_start_end_idx_from_offsets(idx_at_word_ids_list, offsets)
                        idx += len(idx_at_word_ids_list) - 1
                    
                    # Grab all the tokens labeled with I-label
                    while (
                        idx + 1 < len(predictions) and 
                        labelTOtags[predictions[idx + 1]] == f"I_{label}"
                    ):
                        _, end = offsets[idx + 1]
                        idx += 1

                    start, end = start.item(), end.item()
                    word = content[start:end]
                    pred_label[label] += f'{word}、'

                pred_label['content'] = content
                pred_label['decode_sequence'] = predictions
                    # pred_label.append(
                    #     {
                    #         "entity_group": label,
                    #         "word": word,
                    #         "start": start,
                    #         "end": end,
                    #     }
                    # )
                idx += 1
        return pred_label
                
    @staticmethod
    def find_indices(list_to_check, item_to_find) -> List[int]:
        array = np.array(list_to_check)
        indices = np.where(array == item_to_find)[0]
        return list(indices)
    
    def _adjust_start_end_idx_from_offsets(self, idx_list: List[int], offsets) -> Tuple:
       
        """
        Adjust the start and end indices from the offsets list based on the indices list.

        Args:
            idx_list : List[int]
                A list of indices from the offsets list.
            offsets : List[int]
                A list of offsets.

        Returns:
            Tuple[int, int]
                A tuple containing the start and end indices after adjustment.
        
        Example:
            context: 德國 homedics ; tokenize : ['德', '國', 'home', '##di', '##cs']
            offsets: tensor([[ 0,  1],
                            [ 1,  2],
                            [ 3,  7],
                            [ 7,  9],
                            [9, 11]))
            idx_list: [2, 3, 4]
            return: (3, 9)
            start idx of offsets must be with 'home' token and end idx of offsets must be with '##cs' token from the 'homedics' context
        
        """
        start, *_, end = idx_list
        start_token_offset, *_, end_token_offset = offsets[start: end + 1]

        return start_token_offset[0], end_token_offset[-1]