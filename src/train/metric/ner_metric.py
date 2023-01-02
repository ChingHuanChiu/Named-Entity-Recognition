
from typing import Union



from src.train.abstract_class.metric import AbcMetric, ICounter



class NerMetric(AbcMetric):
    def __init__(self, counter: ICounter) -> None:
        self.counter = counter()
        
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.gold_num = 0
        self.pred_num = 0
        self.correct_num = 0
        
        
    def calculate_metric(self, y_batch, y_pred) -> None:
        """
        calculate the all gold_num, pred_num, correct_num in an epoch
        @param y_batch: true label, shape: [batch size, max_seq_length]
        @param y_pred: shape: [batch size, max_seq_length]
        """  

        for pred, true in zip(y_pred, y_batch):

            self.gold_num += self.counter.count_gold_num(true)
            self.pred_num += self.counter.count_pred_num(pred)
            self.correct_num += self.counter.count_correct_num(true, pred)
            
    

        # add instance attribute for the 'get_result' method
        self.precision = self._precision(self.correct_num, self.pred_num)
        self.recall = self._recall(self.gold_num, self.correct_num)
        self.f1 = self._f1_score(self.precision, self.recall)
        
        
    
    def reset(self):

        for name, metric in self.__dict__.items():
            
            if not isinstance(metric, ICounter):
                self.__dict__[name] = 0


    def _precision(self, correct_num: int, predict_num: int) -> Union[float, int]:
        
        if predict_num != 0:
            return correct_num / predict_num
        
        return 0


    def _recall(self, gold_num: int, correct_num: int) -> Union[float, int]:
        if gold_num != 0:
            return correct_num / gold_num
        
        return 0


    def _f1_score(self, precision: float, recall: float) -> float:
        
        if precision + recall != 0:
            score = (2*precision*recall) / (precision + recall)
            return score

        return 0


    




