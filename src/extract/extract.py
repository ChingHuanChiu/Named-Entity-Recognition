from typing import List
from abc import ABCMeta, abstractmethod

from src.model.config import SEQ_MAX_LENGTH 

class IExtract(metaclass=ABCMeta):

    @abstractmethod
    def extract(self, *arg, **kwargs):
        raise NotImplemented("not implemented !")



class BIOExtracter(IExtract):


    def extract(self, content: str, tag: List[str], entity_type) -> List[str]:
        begin_tag = f'B_{entity_type}'
        internal_tag = f'I_{entity_type}'

        tag_length = len(tag)
        content_length = len(content)
        entity_res = []
        
        
        for i in range(content_length):



            if tag[i] == begin_tag:
                
                if i + 1 < content_length:

                    for j in range(i + 1, content_length):
                        
                        if tag[j] == internal_tag:
                            entity_res.append(content[i: j+1])
                            

                        else:
                            # single entity
                            if tag[j - 1] != internal_tag:
                                entity_res.append(content[i])
                            else:
                                
                                break
                else:
                    entity_res.append(content[i])
                    break
                    



        return entity_res



class BIOESExtracter(IExtract):


    def extract(self, content: str, tag: List[str], entity_type) -> List[str]:
        begin_tag = f'B_{entity_type}'
        internal_tag = f'I_{entity_type}'
        end_tag = f'E_{entity_type}'
        single_tag = f'S_{entity_type}'
    
        # SEQ_MAX_LENGTH  - 1 -> remove ['SEP'] tag
        content_length = len(tag)
        entity_res = []
        
        
        for i in range(content_length):

            if tag[i] == single_tag:
                entity_res.append(content[i])
                continue
                
            if tag[i] == begin_tag and i+1 < content_length:
                
                for j in range(i + 1, content_length):
                    if tag[j] == end_tag:
                        entity_res.append(content[i: j+1])
                        break

                    elif tag[j] == internal_tag:
                        continue

                    else:
                        break
        return entity_res
   