from typing import Union

from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import precision_score
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

from src.train.abstract_class.metric import AbcMetric
from src.model.config import labelTOtags


class BIONerMetric(AbcMetric):
    
    def __init__(self) -> None:
        
        self.accuracy = 0
        self.precision = 0
        self.f1_score = 0
        self.recall = 0
        
    def calculate_metric(self, y_true, y_pred) -> None:
        
        y_true = y_true.numpy().tolist()
        y_true = [list(map(labelTOtags.get, _list)) for _list in y_true]
        
        y_pred = y_pred.numpy().tolist()
        y_pred = [list(map(labelTOtags.get, _list)) for _list in y_pred]
        
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred, mode='strict', scheme=IOB2)
        self.recall = recall_score(y_true, y_pred, mode='strict', scheme=IOB2)
        self.f1_score = f1_score(y_true, y_pred, mode='strict', scheme=IOB2)
      
    def reset(self) -> None:
        
        for name, metric in self.__dict__.items():
            
            self.__dict__[name] = 0