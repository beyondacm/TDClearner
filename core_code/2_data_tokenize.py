from utils import *
from Bert_MLP import Config


class Data_Tokenize(object):
    
    def __init__(self):
        print("Loading BERT Model...")
        self.bert_model, \
        self.tokenizer = self.load_BERT()
        print("BERT Loaded!")

        print('Loading Train Dataset...')
        self.train_code_change_lst, \
        self.train_todo_comment_lst, \
        self.train_commit_msg_lst, \
        self.train_label_lst = self.load_data_lst('./data/train_data/cc_todo_pairs.train')

        print("Bert Train Data Encoding ...")
        # Pointwise encoding 
        # self.train_encoded_code_change = self.bert_encode(self.train_code_change_lst)
        # self.train_encoded_todo_comment = self.bert_encode(self.train_todo_comment_lst)
        # self.train_encoded_commit_msg = self.bert_encode(self.train_commit_msg_lst)

        # Pairwise encoding 
        self.train_encoded_code_change = self.bert_encode_pair(self.train_code_change_lst, \
                                                                self.train_todo_comment_lst)
        self.train_encoded_todo_comment = self.bert_encode_pair(self.train_todo_comment_lst, \
                                                                self.train_commit_msg_lst) 
        self.train_encoded_commit_msg = self.bert_encode_pair(self.train_commit_msg_lst, \
                                                                self.train_code_change_lst) 

        print('Loading Val Dataset...')
        self.val_code_change_lst, \
        self.val_todo_comment_lst, \
        self.val_commit_msg_lst, \
        self.val_label_lst = self.load_data_lst('./data/val_data/cc_todo_pairs.val')
 
        self.val_encoded_code_change = self.bert_encode_pair(self.val_code_change_lst, \
                                                                self.val_todo_comment_lst)
        self.val_encoded_todo_comment = self.bert_encode_pair(self.val_todo_comment_lst, \
                                                                self.val_commit_msg_lst) 
        self.val_encoded_commit_msg = self.bert_encode_pair(self.val_commit_msg_lst, \
                                                                self.val_code_change_lst) 


        print('Loading Test Dataset...')
        self.test_code_change_lst, \
        self.test_todo_comment_lst, \
        self.test_commit_msg_lst, \
        self.test_label_lst = self.load_data_lst('./data/test_data/cc_todo_pairs.test')
    
        print("Bert Test Data Encoding ...")
        
        # Pairwise Encodding  
        self.test_encoded_code_change = self.bert_encode_pair(self.test_code_change_lst, \
                                                                self.test_todo_comment_lst)
        self.test_encoded_todo_comment = self.bert_encode_pair(self.test_todo_comment_lst, \
                                                                self.test_commit_msg_lst) 
        self.test_encoded_commit_msg = self.bert_encode_pair(self.test_commit_msg_lst, \
                                                                self.test_code_change_lst) 


    def load_data_lst(self, data_path): 
        code_change_lst = []
        todo_comment_lst = []
        commit_msg_lst = []
        label_lst = []
        with open(data_path, 'r') as fin:
            for line in fin:
                line_split = line.strip().split('\t')
                # code_change, todo_comment, commit_msg, label = line.strip().split('\t')
                code_change = line_split[0]
                todo_comment = line_split[1]
                commit_msg = line_split[2]
                label = line_split[3] 
                label = int(label)
                code_change_lst.append( code_change )
                todo_comment_lst.append( todo_comment )
                commit_msg_lst.append( commit_msg )
                label_lst.append( label )
        return code_change_lst, todo_comment_lst, commit_msg_lst, label_lst

    def load_BERT(self):
        '''
        '''
        ## Tokenize & Input Formatting
        ## Import model/tokenizer 
        ## Load the BERT model
        bert_model = BertModel.from_pretrained('./Model/')
        bert_model.cuda()
        # print("Loading BERT Tokenizer...")
        # tokenizer = AutoTokenizer.from_pretrained('./Model/')
        tokenizer = BertTokenizer.from_pretrained('./Model/')
        return bert_model, tokenizer
  
    def bert_encode(self, input_lst): 
        # bert encoding 
        encoded_input = self.tokenizer(input_lst, padding=True, truncation=True, max_length=128, return_tensors='pt')
        return encoded_input
    
    def bert_encode_pair(self, input_lst1, input_lst2):
        encoded_input_pair = self.tokenizer(input_lst1, input_lst2, \
                                        padding=True, truncation=True, \
                                        max_length=128, return_tensors='pt')
        return encoded_input_pair

    def save(self):
        '''
        save encoded dataset
        '''
        train_encoded_data = (self.train_encoded_code_change, self.train_encoded_todo_comment, self.train_encoded_commit_msg)
        train_labels = np.asarray(self.train_label_lst)

        with open('./data/train_data/train_encoded_data.pkl', 'wb') as handler:
            pickle.dump(train_encoded_data, handler)
       
        with open('./data/train_data/train_labels.pkl', 'wb') as handler:
            pickle.dump(train_labels, handler)

        val_encoded_data = (self.val_encoded_code_change, self.val_encoded_todo_comment, self.val_encoded_commit_msg)
        val_labels = np.asarray(self.val_label_lst)

        with open('./data/val_data/val_encoded_data.pkl', 'wb') as handler:
            pickle.dump(val_encoded_data, handler)
       
        with open('./data/val_data/val_labels.pkl', 'wb') as handler:
            pickle.dump(val_labels, handler)

        test_encoded_data = (self.test_encoded_code_change, self.test_encoded_todo_comment, self.test_encoded_commit_msg)
        test_labels = np.asarray(self.test_label_lst)

        with open('./data/test_data/test_encoded_data.pkl', 'wb') as handler:
            pickle.dump(test_encoded_data, handler)
       
        with open('./data/test_data/test_labels.pkl', 'wb') as handler:
            pickle.dump(test_labels, handler)


def main():
    
    dt = Data_Tokenize()
    dt.save()
    pass

if __name__ == '__main__':
    main()

