from operator import itemgetter 
from typing import List, Tuple


import numpy as np
import torch

from src.model.config import B, I, O, SpecialToken
from src.train.abstract_class.metric import ICounter






class BIONerCounter(ICounter):

    
    def __init__(self) -> None:


        self.idx_of_entity_B =  [int(b.name.replace('_', '')) for b in B]


    def count_gold_num(self, y_true) -> int:
        """Count the number of 'B' entity  in true label which represent the number of entites in y_true
        """

        count = 0

        idx_of_total_entity = self.idx_of_entity_B

        unique_y, y_true_label_count = torch.unique(y_true, return_counts=True)

        for n in idx_of_total_entity:
            if n in unique_y:
                index_of_class = list(unique_y.numpy()).index(n)
                count += y_true_label_count[index_of_class].numpy()

        return count


    def count_pred_num(self, y_pred) -> int:

        count = 0
        y_pred = list(y_pred)
        thing_b_idx_in_y_pred = self._find_indices(y_pred, 2)
        
        thing_b_idx_in_y_pred, thing_single_entity_indice = self._adjust_B_tag_entity_indices(y_pred, thing_b_idx_in_y_pred, o_tag_index=1, i_tag_index=3)

        person_b_idx_in_y_pred = self._find_indices(y_pred, 4)
        person_b_idx_in_y_pred, person_single_entity_indice = self._adjust_B_tag_entity_indices(y_pred, person_b_idx_in_y_pred, o_tag_index=1, i_tag_index=5)
        
        location_b_idx_in_y_pred = self._find_indices(y_pred, 6)
        location_b_idx_in_y_pred, location_single_entity_indice = self._adjust_B_tag_entity_indices(y_pred,location_b_idx_in_y_pred, o_tag_index=1, i_tag_index=7)
        
        time_b_idx_in_y_pred = self._find_indices(y_pred, 8)
        time_b_idx_in_y_pred, time_single_entity_indice = self._adjust_B_tag_entity_indices(y_pred, time_b_idx_in_y_pred, o_tag_index=1, i_tag_index=9)
        
        metric_b_idx_in_y_pred = self._find_indices(y_pred, 10)
        metric_b_idx_in_y_pred, metric_single_entity_indice = self._adjust_B_tag_entity_indices(y_pred, metric_b_idx_in_y_pred, o_tag_index=1, i_tag_index=11)

        organization_b_idx_in_y_pred = self._find_indices(y_pred, 12)
        organization_b_idx_in_y_pred, organization_single_entity_indice = self._adjust_B_tag_entity_indices(y_pred, organization_b_idx_in_y_pred, o_tag_index=1, i_tag_index=13)

        abstract_b_idx_in_y_pred = self._find_indices(y_pred, 14)
        abstract_b_idx_in_y_pred, abstract_single_entity_indice = self._adjust_B_tag_entity_indices(y_pred, abstract_b_idx_in_y_pred, o_tag_index=1, i_tag_index=15)
        
        physical_b_idx_in_y_pred = self._find_indices(y_pred, 16)
        physical_b_idx_in_y_pred, physical_single_entity_indice = self._adjust_B_tag_entity_indices(y_pred, physical_b_idx_in_y_pred, o_tag_index=1, i_tag_index=17)


        term_b_idx_in_y_pred = self._find_indices(y_pred, 18)
        term_b_idx_in_y_pred, term_single_entity_indice = self._adjust_B_tag_entity_indices(y_pred, term_b_idx_in_y_pred, o_tag_index=1, i_tag_index=19)

        count += len(thing_b_idx_in_y_pred) \
               + len(person_b_idx_in_y_pred)\
               + len(location_b_idx_in_y_pred) \
               + len(time_b_idx_in_y_pred) \
               + len(metric_b_idx_in_y_pred) \
               + len(organization_b_idx_in_y_pred)\
               + len(abstract_b_idx_in_y_pred) \
               + len(person_b_idx_in_y_pred) \
               + len(term_b_idx_in_y_pred) \
               + len(thing_single_entity_indice)\
               + len(person_single_entity_indice) \
               + len(location_single_entity_indice) \
               + len(time_single_entity_indice) \
               + len(metric_single_entity_indice) \
               + len(organization_single_entity_indice) \
               + len(abstract_single_entity_indice) \
               + len(physical_single_entity_indice) \
               + len(term_single_entity_indice)            

        return count
    
    
    def count_correct_num(self, y_true, y_pred):
        """
        @param y_true: class id  of true label, which is tensor , can not be one-hot type
        @param y_pred: class id  of predict label, which is tensor , can not be one-hot type
        """
        # 0 indicate the correct prediction
        y_true_substract_y_pred = list(y_true.numpy() - y_pred.numpy())
        y_true_list, y_pred_list = list(y_true.numpy()), list(y_pred.numpy())
        
        thing_b_idx_in_y_true = self._find_indices(y_true_list, 2)

        person_b_idx_in_y_true = self._find_indices(y_true_list, 4)

        location_b_idx_in_y_true = self._find_indices(y_true_list, 6)

        time_b_idx_in_y_true = self._find_indices(y_true_list, 8)
        # time_b_idx_in_y_true, time_single_entity_indice = self._adjust_B_tag_entity_indices(y_true_list, time_b_idx_in_y_true, o_tag_index=1, i_tag_index=9)

        metric_b_idx_in_y_true = self._find_indices(y_true_list, 10)

        organization_b_idx_in_y_true = self._find_indices(y_true_list, 12)

        abstract_b_idx_in_y_true = self._find_indices(y_true_list, 14)

        
        physical_b_idx_in_y_true = self._find_indices(y_true_list, 16)
       
        term_b_idx_in_y_true = self._find_indices(y_true_list, 18)

 
        count = 0
        count += self._count_correct_prediction(y_true=y_true_list,
                                                y_true_substract_y_pred_list=y_true_substract_y_pred,
                                                begin_entity_idx=thing_b_idx_in_y_true,
                                                i_tag=3) \
                + self._count_correct_prediction(y_true=y_true_list,
                                                y_true_substract_y_pred_list=y_true_substract_y_pred,
                                                begin_entity_idx=person_b_idx_in_y_true,
                                                i_tag=5)  \
                + self._count_correct_prediction(y_true=y_true_list,
                                                y_true_substract_y_pred_list=y_true_substract_y_pred,
                                                begin_entity_idx=location_b_idx_in_y_true,
                                                i_tag=7)  \
                + self._count_correct_prediction(y_true=y_true_list,
                                                y_true_substract_y_pred_list=y_true_substract_y_pred,
                                                begin_entity_idx=time_b_idx_in_y_true,
                                                i_tag=9) \
                + self._count_correct_prediction(y_true=y_true_list,
                                                y_true_substract_y_pred_list=y_true_substract_y_pred,
                                                begin_entity_idx=metric_b_idx_in_y_true,
                                                i_tag=11) \
                + self._count_correct_prediction(y_true=y_true_list,
                                                y_true_substract_y_pred_list=y_true_substract_y_pred,
                                                begin_entity_idx=organization_b_idx_in_y_true,
                                                i_tag=13) \
                + self._count_correct_prediction(y_true=y_true_list,
                                                y_true_substract_y_pred_list=y_true_substract_y_pred,
                                                begin_entity_idx=abstract_b_idx_in_y_true,
                                                i_tag=15) \
                + self._count_correct_prediction(y_true=y_true_list,
                                                y_true_substract_y_pred_list=y_true_substract_y_pred,
                                                begin_entity_idx=physical_b_idx_in_y_true,
                                                i_tag=17) \
                + self._count_correct_prediction(y_true=y_true_list,
                                                y_true_substract_y_pred_list=y_true_substract_y_pred,
                                                begin_entity_idx=term_b_idx_in_y_true,
                                                i_tag=19)                                                  

       

        return count


    def _find_indices(self, list_to_check, item_to_find) -> List[int]:
        array = np.array(list_to_check)
        indices = np.where(array == item_to_find)[0]
        return list(indices)
    
    

    def _adjust_B_tag_entity_indices(self, y: List[int], b_tag_idx_in_y: List[int], o_tag_index: int, i_tag_index: int) -> Tuple:
        """conform if the "B" tag is the single entity or not and return the index of single entity  and 
        the index of remain 'B' tag  
        """
        single_entity_index = []
        b_tag_idx_in_y_cp = b_tag_idx_in_y.copy()
        for i in b_tag_idx_in_y_cp:
            if (i == (len(y) - 1)) or (i == 0 and y[i + 1] != i_tag_index) or (y[i + 1] != i_tag_index):
                single_entity_index.append(i)
                b_tag_idx_in_y.remove(i)


        return b_tag_idx_in_y, single_entity_index


    def _count_correct_prediction(self, 
                                y_true,
                                y_true_substract_y_pred_list,
                                begin_entity_idx,
                                i_tag: int 
                                ) -> int:
        res_count = 0
        for b_tag_index in begin_entity_idx:
            entity_length = self._get_entity_length(y_true, b_tag_index, i_tag)
            if all(i == 0 for i in y_true_substract_y_pred_list[b_tag_index: b_tag_index + entity_length]):
                res_count += 1
        # print(res_count)
        return res_count
    

    def _get_entity_length(self, y_true: List[int], b_tag_index: int, i_tag: int):
        count = 0
        i = b_tag_index

        while i + 1 < len(y_true):
            if y_true[i + 1] != i_tag:
                
                if y_true[i] != i_tag:
                    count += 1
                    
                else:
                    count += 1
                    i += 1
                break
                       
            else:
                count += 1
                i += 1

        
        return count



# class BIOESNerCounter(ICounter):
#     """count the indicator of prediction by Ner model base on tag level, the following indicator:
#     1. gold_num: the number of entity in true label
#     2. predict_num
#     3. correct_num

    
#     """
#     def __init__(self) -> None:


#         self.idx_of_entity_B =  [int(b.name.replace('_', '')) for b in B]
#         # self.idx_of_entity_S = [int(s.name.replace('_', '')) for s in S]


#     def count_gold_num(self, y_true: tf.Tensor) -> int:
#         """Count the number of 'B' entity and 's' entity in true label.
#         the reason I only count the 'B' entity and 's' entity is that when 'B' or 'S' tag appear 
#         in the label meaning a complete entity. i.g. y_trut = ['B', 'I', 'E', 'O', 'O', 'S'], the numbers of 
#         entity are '2'
#         """

#         count = 0

#         idx_of_total_entity = self.idx_of_entity_B + self.idx_of_entity_S

#         y_true_label_count = tf.unique_with_counts(y_true)

#         for n in idx_of_total_entity:
#             if n in y_true_label_count.y:
#                 index_of_class = list(y_true_label_count.y.numpy()).index(n)
#                 count += y_true_label_count.count[index_of_class].numpy()

#         return count


#     def count_pred_num(self, y_pred: tf.Tensor) -> int:
#         count = 0
#         y_pred = list(y_pred)
#         brand_b_idx_in_y_pred = self._find_indices(y_pred, 2)
#         brand_e_idx_in_y_pred = self._find_indices(y_pred, 4)
#         brand_s_idx_in_y_pred = self._find_indices(y_pred, 5)


#         type_b_idx_in_y_pred = self._find_indices(y_pred, 6)
#         type_e_idx_in_y_pred = self._find_indices(y_pred, 8)
#         type_s_idx_in_y_pred = self._find_indices(y_pred, 9)
 
#         # print(self._find_entity_count(y_pred, brand_b_idx_in_y_pred, brand_e_idx_in_y_pred, 3))
#         # print(self._find_entity_count(y_pred, type_b_idx_in_y_pred, type_e_idx_in_y_pred, 7))
#         count += len(brand_s_idx_in_y_pred) \
#                + len(type_s_idx_in_y_pred) \
#                + self._find_entity_count(y_pred, brand_b_idx_in_y_pred, brand_e_idx_in_y_pred, 3) \
#                + self._find_entity_count(y_pred, type_b_idx_in_y_pred, type_e_idx_in_y_pred, 7)
#         return count
    
    
#     def count_correct_num(self, y_true: tf.Tensor, y_pred: tf.Tensor):
#         """
#         @param y_true: class id  of true label, which is tensor , can not be one-hot type
#         @param y_pred: class id  of predict label, which is tensor , can not be one-hot type
#         """
#         # 0 indicate the correct prediction
#         y_true_substract_y_pred = list(y_true.numpy() - y_pred.numpy())
#         y_true_list, y_pred_list = list(y_true), list(y_pred)
        
#         brand_b_idx_in_y_true = self._find_indices(y_true_list, 2)
#         brand_e_idx_in_y_true = self._find_indices(y_true_list, 4)
#         brand_s_idx_in_y_true = self._find_indices(y_true_list, 5)


#         type_b_idx_in_y_true = self._find_indices(y_true_list, 6)
#         type_e_idx_in_y_true = self._find_indices(y_true_list, 8)
#         type_s_idx_in_y_true = self._find_indices(y_true_list, 9)
#         count = 0
#         count += self._count_correct_prediction(y_true_substract_y_pred_list=y_true_substract_y_pred,
#                                                 begin_entity_idx=brand_b_idx_in_y_true,
#                                                 end_entity_idx=brand_e_idx_in_y_true) + \
#                 self._count_correct_prediction(y_true_substract_y_pred_list=y_true_substract_y_pred,
#                                                 begin_entity_idx=type_b_idx_in_y_true,
#                                                 end_entity_idx=type_e_idx_in_y_true) +\
#                 self._count_correct_single_tag_prediction(y_true_substract_y_pred_list=y_true_substract_y_pred,
#                                                         s_idx_in_y_true=brand_s_idx_in_y_true
#                                                         )   +\
#                 self._count_correct_single_tag_prediction(y_true_substract_y_pred_list=y_true_substract_y_pred,
#                                                         s_idx_in_y_true=type_s_idx_in_y_true
#                                                         )

#         return count


#     def _find_indices(self, list_to_check, item_to_find):
#         array = np.array(list_to_check)
#         indices = np.where(array == item_to_find)[0]
#         return list(indices)


#     def _find_entity_count(self, y, begin_entity_idx, end_entity_idx, i_tag_idx) -> int:
#         if len(begin_entity_idx) == 0 or len(end_entity_idx) == 0:
#             return 0
        
#         entity_count = 0
#         for start in begin_entity_idx:

#             for end in end_entity_idx:
#                 if start > end :
#                     continue

#                 if end - start == 1:
#                     entity_count += 1
#                     break

#                 tag_between_start_end = y[start + 1: end]
                

#                 # if not all tag is 'I' between 'B' tag and 'E' tag
#                 if not all(i == i_tag_idx for i in tag_between_start_end):
#                     break
#                 else:
#                     entity_count += 1 
#         return entity_count


#     def _count_correct_prediction(self, 
#                                 y_true_substract_y_pred_list,
#                                 begin_entity_idx, 
#                                 end_entity_idx
#                                 ) -> int:
#         res_count = 0
#         for start, end in zip(begin_entity_idx, end_entity_idx):
            
#             if all(i == 0 for i in y_true_substract_y_pred_list[start: end + 1]):
#                 res_count += 1
#         return res_count


#     def _count_correct_single_tag_prediction(self, 
#                                         y_true_substract_y_pred_list,
#                                         s_idx_in_y_true
#                                         ) -> int:

#         if len(s_idx_in_y_true) == 0:
#             return 0

#         predict_result = tuple(y_true_substract_y_pred_list[i] for i in s_idx_in_y_true)
#         return sum(_i == 0 for _i in predict_result)
    
