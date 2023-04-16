from typing import Optional

import numpy as np
import torch
from torch import nn
from transformers import BertModel
from torchcrf import CRF
import torch.nn.functional as F



from src.model.config import HUGGINGFACE_MODEL, SEQ_MAX_LENGTH


class NERBertWithCRF(nn.Module):
    

    def __init__(self, num_label: int, hidden_dropout_prob=0.2) -> None:
        super().__init__()
        self.num_label = num_label
 
        self.bert = BertModel.from_pretrained(HUGGINGFACE_MODEL, output_attentions=True)
        
        self.dropout = nn.Dropout(hidden_dropout_prob)
 
        self.logits = nn.Linear(768, self.num_label)
        self.crf = CRF(self.num_label, batch_first=True)


    def forward(self, inputs_dict, y_true=None):

        mask = inputs_dict["attention_mask"]
        
        mask = mask.bool()
        

        last_hidden_state = self.bert(**inputs_dict).last_hidden_state
        
        last_hidden_state = self.dropout(last_hidden_state)

        logits = self.logits(last_hidden_state)

        
        decode_sequence = self.crf.decode(emissions=logits, mask=mask)


        if y_true is not None:
            
            crf_log_likelihood = self.crf(emissions=logits, tags=y_true, mask=mask, reduction='mean')
            
            return crf_log_likelihood, decode_sequence
            
        return  decode_sequence

    
    def init_weights(self):

        for m in self.logits.modules():
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()




class NERBertBiLSTMWithCRF(nn.Module):

    def __init__(self, 
                 num_label: int, 
                 lstm_num_layers: int, 
                 local_rank:int,
                 device: str) -> None:

        super().__init__()
        self.num_label = num_label
        self.local_rank = local_rank
        self.device = device
        self.bert = BertModel.from_pretrained(HUGGINGFACE_MODEL, output_attentions=True)
            
        
        self.num_layers = lstm_num_layers
        self.bi_lstm = nn.LSTM(
                            input_size=768, 
                            hidden_size=128,
                            num_layers=lstm_num_layers, 
                            batch_first=True, 
                            bidirectional=True)

        
        self.linear = nn.Linear(256, num_label)
        self.dropout = nn.Dropout(p=0.1)

        self.crf = CRF(num_label, batch_first=True)

        

    def forward(self, inputs_dict, y_true=None):
        mask = inputs_dict["attention_mask"]
        mask = mask.bool()

        last_hidden_state = self.bert(**inputs_dict).last_hidden_state


        
        h_0 = torch.randn(2 * self.num_layers, last_hidden_state.size()[0], 128).to(torch.device(self.device, self.local_rank)) #shape: (num_direction * num_layers, bs, hidden_size)
        c_0 = torch.randn(2 * self.num_layers, last_hidden_state.size()[0], 128).to(torch.device(self.device, self.local_rank))


        last_hidden_state, (h,c) = self.bi_lstm(last_hidden_state, (h_0, c_0))
        
        logits = self.linear(self.dropout(F.relu(last_hidden_state)))
    
        
        decode_sequence = self.crf.decode(emissions=logits, mask=mask)

        if y_true is not None:
            crf_log_likelihood = self.crf(emissions=logits, tags=y_true, mask=mask, reduction='mean')
            
            return crf_log_likelihood, decode_sequence
            


        return  decode_sequence

    
    def init_weights(self):

        for m in self.linear.modules():
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.bi_lstm.modules():
            for n, p in m.named_parameters():
                    if 'weight' in n:
                        nn.init.xavier_normal_(p)
                    elif 'bias' in n:
                        nn.init.zeros_(p)
                        
                        
                        
class NERBertCNNWithCRF(nn.Module):

    def __init__(self, num_label: int, local_rank:int) -> None:
        super().__init__()
        self.num_label = num_label
        self.local_rank = local_rank
        self.bert = BertModel.from_pretrained(HUGGINGFACE_MODEL, output_attentions=True)
            
        
        #https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        filter_sizes = [5] 
        num_filters = [256] 
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=768,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i],
                      padding='same'
                     )
            for i in range(len(filter_sizes))
        ])
        
        
        
        self.linear = nn.Linear(np.sum(num_filters), num_label)
  

        self.crf = CRF(num_label, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, inputs_dict, y_true=None):
        mask = inputs_dict["attention_mask"]
        mask = mask.bool()

        last_hidden_state = self.bert(**inputs_dict).last_hidden_state

  
       
        x_reshaped = last_hidden_state.permute(0, 2, 1)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        # x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]
        
        x_fc = torch.cat([conv.permute(0, 2, 1) for conv in x_conv_list], dim=-1)

        logits = self.linear(self.dropout(x_fc))

        decode_sequence = self.crf.decode(emissions=logits, mask=mask)

        
        if y_true is not None:
            crf_log_likelihood = self.crf(emissions=logits, tags=y_true, mask=mask, reduction='mean')
            
            return crf_log_likelihood, decode_sequence
            


        return  decode_sequence

    
    def init_weights(self):

        for m in self.children():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)



 


        





