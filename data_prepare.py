from typing import List

import pandas as pd
import numpy as np




def convert_textfile_to_df(datapath: str, max_length: int) -> pd.DataFrame:
    """convert the text file data to dataframe and also split the data 
    if the length of context is longer than SEQ_MAX_LENGTH
    
    """
    MAP_DICT = {"B_Thing": 2, "B_Person": 4, "B_Location": 6, "B_Time": 8, "B_Metric": 10, "B_Organization": 12, "B_Abstract": 14, "B_ABstract": 14, "B_Physical":16,"B_Term": 18,
                "I_Thing": 3, "I_Person": 5, "I_Location": 7, "I_Time": 9, "I_Metric": 11, "I_Organization": 13, "I_Abstract": 15, "I_ABstract": 15, "I_Physical": 17, "I_Term": 19,
                "O": 1, "[PAD]": 0, "[SEP]": 20
                }
    # exclude [CLS] ans [SEP] tokens            
    max_length = max_length -2
    with open(datapath, 'r') as f:
        data = f.readlines()
        res_dict = dict()
        context, tag, tag_id = '', [], []
        context_result, tag_result, tag_id_result = [], [], []
        for i in data:
            split = i.split(' ')
            if len(split) != 2 and split[0] != '\n':
                print('error encounter', i)
                continue
            
            if split[0] != '\n':
                
                context += split[0]
                t = split[1].strip()
                tag.append(t)
                tag_id.append(MAP_DICT[t])
                
                if split[0] in ['，', '。', '!', '?'] and len(context) <= max_length :
                    breakpoint_flag_idx = len(context)
                    
                if len(context) > max_length:
                    sub_context = context[: breakpoint_flag_idx]
                    sub_tag = tag[: breakpoint_flag_idx]
                    sub_tag_id = tag_id[: breakpoint_flag_idx]
                    assert len(sub_context) == len(sub_tag)
                    
                    context_result.append(sub_context)
                    tag_result.append(sub_tag)
                    tag_id_result.append(sub_tag_id)
                    
                    
                    context = context[breakpoint_flag_idx: ]
                    tag = tag[breakpoint_flag_idx: ]
                    tag_id = tag_id[breakpoint_flag_idx: ]

                    
                assert len(context) == len(tag)

                

            else:

                context_result.append(context)
                tag_result.append(tag)

                tag_id_result.append(tag_id)
                context, tag, tag_id = '', [], []

        res_dict['context'] = context_result
        res_dict['tag'] = tag_result
        res_dict['tag_id'] = tag_id_result
        res_df = pd.DataFrame(res_dict)
    return res_df



