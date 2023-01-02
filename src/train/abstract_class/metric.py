from abc import ABCMeta, abstractmethod

from typing import Dict, Union


                    
class AbcMetric(metaclass=ABCMeta):

    @abstractmethod
    def calculate_metric(self, **kwargs) -> None:
        raise NotImplemented("not implemented")


    @abstractmethod
    def reset(self):
        raise NotImplemented("not implemented")


    def get_result(self) -> Dict[str, Union[float, int]]:
        """
        return a dictionary of result of metrics
        """
        result = dict()
        for name, metric in self.__dict__.items():

            result[name] = metric

        return result
    
    

class ICounter(metaclass=ABCMeta):

    @abstractmethod
    def count_gold_num(slef, y_true):
        pass

    @abstractmethod
    def count_pred_num(self, y_pred):
        pass

    @abstractmethod
    def count_correct_num(y_true, y_pred):
        pass