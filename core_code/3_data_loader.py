from utils import *
from Bert_MLP import Config
import numpy as np

class Data_Loader(object):
    
    def __init__(self):
        self.config = Config()
        self.batch_size = self.config.batch_size 
        print("batch_size:", self.batch_size)
        
        # Load Encoded Train Data 
        self.train_encoded_data = self.load_encoded_data('./data/train_data/train_encoded_data.pkl')
        self.train_encoded_code_change, \
        self.train_encoded_todo_comment, \
        self.train_encoded_commit_msg = self.train_encoded_data
        self.train_labels = self.load_encoded_data('./data/train_data/train_labels.pkl')
        
        self.train_cc = self.make_train_data(self.train_encoded_code_change)      
        self.train_todo = self.make_train_data(self.train_encoded_todo_comment)      
        self.train_msg = self.make_train_data(self.train_encoded_commit_msg)      

        # Val Data
        self.val_encoded_data = self.load_encoded_data('./data/val_data/val_encoded_data.pkl')
        self.val_encoded_code_change, \
        self.val_encoded_todo_comment, \
        self.val_encoded_commit_msg = self.val_encoded_data
        self.val_labels = self.load_encoded_data('./data/val_data/val_labels.pkl')
        
        self.val_cc = self.make_val_data(self.val_encoded_code_change)      
        self.val_todo = self.make_val_data(self.val_encoded_todo_comment)      
        self.val_msg = self.make_val_data(self.val_encoded_commit_msg)      


        ############################
        # Load Raw Test Data 
        ############################

        # Do the same thing for the Test Set  
        self.test_encoded_data = self.load_encoded_data('./data/test_data/test_encoded_data.pkl')
        self.test_encoded_code_change, \
        self.test_encoded_todo_comment, \
        self.test_encoded_commit_msg = self.test_encoded_data
        self.test_labels = self.load_encoded_data('./data/test_data/test_labels.pkl')

        self.test_cc = self.make_test_data(self.test_encoded_code_change)      
        self.test_todo = self.make_test_data(self.test_encoded_todo_comment)      
        self.test_msg = self.make_test_data(self.test_encoded_commit_msg)      
        
        # make dataloader 
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.make_dataloader()
    
    def load_data_lst(self, data_path):
        '''
        '''
        code_change_lst = []
        todo_comment_lst = []
        commit_msg_lst = []
        label_lst = []
        with open(data_path, 'r') as fin:
            for line in fin:
                code_change, todo_comment, commit_msg, label = line.strip().split('\t')
                label = int(label)
                code_change_lst.append( code_change )
                todo_comment_lst.append( todo_comment )
                commit_msg_lst.append( commit_msg )
                label_lst.append( label )
        return code_change_lst, todo_comment_lst, commit_msg_lst, label_lst


    def load_encoded_data(self, data_path):
        with open(data_path, 'rb') as handler:
            encoded_data= pickle.load(handler)
        return encoded_data

    def load_labels(self, data_path):
        with open(data_path, 'rb') as handler:
            labels = pickle.load(handler)
        return labels
    
    def make_train_data(self, encoded_data):      
        input_ids, \
        token_type_ids, \
        attention_masks = encoded_data['input_ids'], encoded_data['token_type_ids'], encoded_data['attention_mask']
        # Convert to Pytorch Data Types
        inputs = torch.tensor(input_ids)
        types = torch.tensor(token_type_ids)
        masks = torch.tensor(attention_masks)
        labels = torch.tensor(self.train_labels)
        train_data = (inputs, masks, types, labels)
        return train_data

    def make_val_data(self, encoded_data):      
        input_ids, \
        token_type_ids, \
        attention_masks = encoded_data['input_ids'], encoded_data['token_type_ids'], encoded_data['attention_mask']
        # Convert to Pytorch Data Types
        inputs = torch.tensor(input_ids)
        types = torch.tensor(token_type_ids)
        masks = torch.tensor(attention_masks)
        labels = torch.tensor(self.val_labels)
        val_data = (inputs, masks, types, labels)
        return val_data

    def make_test_data(self, encoded_data):      
        input_ids, \
        token_type_ids, \
        attention_masks = encoded_data['input_ids'], encoded_data['token_type_ids'], encoded_data['attention_mask']
        # Convert to Pytorch Data Types
        inputs = torch.tensor(input_ids)
        types = torch.tensor(token_type_ids)
        masks = torch.tensor(attention_masks)
        labels = torch.tensor(self.test_labels)
        test_data = (inputs, masks, types, labels)
        return test_data
    
    def make_dataloader(self):
        '''
        '''
        # make the train_dataloader 
        train_data = TensorDataset(self.train_cc[0], self.train_cc[1], self.train_cc[2],\
                                    self.train_todo[0], self.train_todo[1], self.train_todo[2], \
                                    self.train_msg[0], self.train_msg[1], self.train_msg[2], \
                                    self.train_cc[3]
                                    )
        # train_sampler = RandomSampler(train_data)
        # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size)

        # make the val_dataloader 
        val_data = TensorDataset(self.val_cc[0], self.val_cc[1], self.val_cc[2],\
                                    self.val_todo[0], self.val_todo[1], self.val_todo[2], \
                                    self.val_msg[0], self.val_msg[1], self.val_msg[2], \
                                    self.val_cc[3]
                                    )
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size)


        # make the test_dataloader 
        test_data = TensorDataset(self.test_cc[0], self.test_cc[1], self.test_cc[2],\
                                    self.test_todo[0], self.test_todo[1], self.test_todo[2], \
                                    self.test_msg[0], self.test_msg[1], self.test_msg[2], \
                                    self.test_cc[3] 
                                    )
        # test_sampler = RandomSampler(test_data)
        # test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size)
        return train_dataloader, val_dataloader, test_dataloader
        pass

    def save_dataloader(self):
        '''
        '''
        with open('./data/train_data/train_dataloader.pkl', 'wb') as handler:
            pickle.dump(self.train_dataloader, handler) 

        with open('./data/val_data/val_dataloader.pkl', 'wb') as handler:
            pickle.dump(self.val_dataloader, handler) 
        
        with open('./data/test_data/test_dataloader.pkl', 'wb') as handler:
            pickle.dump(self.test_dataloader, handler) 
        pass


def main():
    
    # ds = Data_Split()
    dt = Data_Loader()
    dt.save_dataloader()
    pass

if __name__ == '__main__':
    main()
