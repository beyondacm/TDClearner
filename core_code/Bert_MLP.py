from utils import *

class Config(object):

    def __init__(self): 
        self.model_name = 'bert'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.num_classes = 2
        self.bert_path = './Model'
        self.hidden_size = 768
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.batch_size = 16
        self.num_epochs = 5 


class Model(nn.Module):
    
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True 
        self.fc0 = nn.Linear(3*config.hidden_size, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, config.num_classes)

    # def forward(self, input_ids, attention_mask, token_type_ids):
    def forward(self, cc_input, td_input, msg_input):
         
        cc_input_ids, cc_input_mask, cc_input_types = cc_input[0], cc_input[1], cc_input[2] 
        td_input_ids, td_input_mask, td_input_types = td_input[0], td_input[1], td_input[2] 
        msg_input_ids, msg_input_mask, msg_input_types = msg_input[0], msg_input[1], msg_input[2] 

        # _, cc_pooled = self.bert(input_ids = cc_input_ids, \
        #                                attention_mask = cc_input_mask, \
        #                                token_type_ids = cc_input_types) 

        # _, td_pooled = self.bert(input_ids = td_input_ids, \
        #                                attention_mask = td_input_mask, \
        #                                token_type_ids = td_input_types) 

        cc_outputs = self.bert(input_ids = cc_input_ids, \
                               attention_mask = cc_input_mask, \
                               token_type_ids = cc_input_types) 


        td_outputs = self.bert(input_ids = td_input_ids, \
                               attention_mask = td_input_mask, \
                               token_type_ids = td_input_types) 
        
        msg_outputs = self.bert(input_ids = msg_input_ids, \
                               attention_mask = msg_input_mask, \
                               token_type_ids = msg_input_types) 
            
        cc_pooled = cc_outputs.pooler_output 
        td_pooled = td_outputs.pooler_output
        msg_pooled = msg_outputs.pooler_output

        features = torch.cat((cc_pooled, td_pooled, msg_pooled), dim=1)
        features = self.fc0(features)
        features = self.fc1(features)
        out = self.fc2(features)
        return out

