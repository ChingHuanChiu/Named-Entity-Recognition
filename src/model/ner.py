from typing import Optional
from unicodedata import bidirectional, name

import torch
from torch import nn
from transformers import BertModel
from torchcrf import CRF


from src.model.config import HUGGINGFACE_MODEL, SEQ_MAX_LENGTH


class NERBertWithCRF(nn.Module):
    

    def __init__(self, num_label: int, freeze_bert: bool) -> None:
        super().__init__()
        self.num_label = num_label
        self.freeze_bert = freeze_bert

 
        self.bert = BertModel.from_pretrained(HUGGINGFACE_MODEL, output_attentions=True)
        
        if self.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        
 
        self.logits = nn.Linear(768, self.num_label)
        self.crf = CRF(self.num_label, batch_first=True)

            


    def forward(self, inputs_dict, y_true=None):

        mask = inputs_dict["attention_mask"]

        last_hidden_state = self.bert(**inputs_dict).last_hidden_state

        # remove [CLS] hidden state
        last_hidden_state = last_hidden_state[:, 1:, :]
        mask = mask[:, 1:].bool()
        # TODO: move to  the unit test
        assert y_true.size()[-1] == SEQ_MAX_LENGTH - 1, "wrong shape in y_true"


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

    def __init__(self, num_label: int, freeze_bert: bool, lstm_num_layers: int, local_rank:int) -> None:
        super().__init__()
        self.num_label = num_label
        self.freeze_bert = freeze_bert
        self.local_rank = local_rank
        self.bert = BertModel.from_pretrained(HUGGINGFACE_MODEL, output_attentions=True)
            
        if self.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        
        self.num_layers = lstm_num_layers
        self.bi_lstm = nn.LSTM(
                            input_size=768, 
                            hidden_size=128,
                            num_layers=lstm_num_layers, 
                            batch_first=True, 
                            bidirectional=True)

        
        self.linear = nn.Linear(256, num_label)

        self.crf = CRF(num_label, batch_first=True)

        

    def forward(self, inputs_dict, y_true=None):
        mask = inputs_dict["attention_mask"]

        last_hidden_state = self.bert(**inputs_dict).last_hidden_state

        # remove [CLS] hidden state
        bert_hidden_state = last_hidden_state[:, 1:, :]
        mask = mask[:, 1:].bool()
        h_0 = torch.randn(2 * self.num_layers, bert_hidden_state.size()[0], 128).to(torch.device("cuda", self.local_rank)) #shape: (num_direction * num_layers, bs, hidden_size)
        c_0 = torch.randn(2 * self.num_layers, bert_hidden_state.size()[0], 128).to(torch.device("cuda", self.local_rank))


        last_hidden_state, (h,c) = self.bi_lstm(bert_hidden_state, (h_0, c_0))
        
        logits = self.linear(last_hidden_state)
        
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
        self.cnn1d = nn.Conv1d(SEQ_MAX_LENGTH - 1, SEQ_MAX_LENGTH - 1, 5, stride=2) # dimension: 382
        self.cnn1d_dilation2 = nn.Conv1d(SEQ_MAX_LENGTH - 1, SEQ_MAX_LENGTH - 1, 3, stride=2, dilation=2)
        
        self.linear = nn.Linear(189, num_label)
  

        self.crf = CRF(num_label, batch_first=True)


    def forward(self, inputs_dict, y_true=None):
        mask = inputs_dict["attention_mask"]

        last_hidden_state = self.bert(**inputs_dict).last_hidden_state

        # remove [CLS] hidden state
        bert_hidden_state = last_hidden_state[:, 1:, :]
        mask = mask[:, 1:].bool()

        
        cnn_hidden_state = self.cnn1d(bert_hidden_state)
        cnn_hidden_state = self.cnn1d_dilation2(cnn_hidden_state)    
        logits = self.linear(cnn_hidden_state)

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

        for m in self.cnn1d.modules():
            for n, p in m.named_parameters():
                    if 'weight' in n:
                        nn.init.xavier_normal_(p)
                    elif 'bias' in n:
                        nn.init.zeros_(p)



 


        





