from utils import *
from Bert_MLP import Model, Config
import torch.nn.functional as F
import torch.nn
import numpy
# from Bert_CNN import Model, Config
# torch.manual_seed(777)

# Load the iterator
with open('./data/train_data/train_dataloader.pkl', 'rb') as handler:
    train_dataloader = pickle.load(handler)
print("train dataloader loaded!")

with open('./data/test_data/test_dataloader.pkl', 'rb') as handler:
    test_dataloader = pickle.load(handler)
print("test dataloader loaded!")

PATH = './model_save/epoch3/model.ckpt'
config = Config()

# Load Model
model = Model(config).to(config.device)
model.load_state_dict(torch.load(PATH))
model.eval()
print('Model Loaded!')

# ========================================
#              Training 
# ========================================
print("")
print("Running Training...")

t0 = time.time()
# Put the model in evaluation mode--the dropout layers behave differently
# during evaluation.

# Tracking variables
total_eval_accuracy = 0
total_eval_loss = 0
nb_eval_steps = 0

prob_result_lst = []

# Evaluate data for one epoch 
for batch in train_dataloader:
    # Unpack this training batch from our dataloader 
    # As we unpack the batch, we will also copy each tensor to the GPU using 'to' method
    # 'batch' contains three pytorch tensors:
    # [0]: input ids
    # [1]: attention masks
    # [2]: labels
    cc_input_ids   = batch[0].to(config.device)
    cc_input_mask  = batch[1].to(config.device)
    cc_input_types = batch[2].to(config.device)
    
    td_input_ids   = batch[3].to(config.device)
    td_input_mask  = batch[4].to(config.device)
    td_input_types = batch[5].to(config.device)

    msg_input_ids   = batch[6].to(config.device)
    msg_input_mask  = batch[7].to(config.device)
    msg_input_types = batch[8].to(config.device)

    b_labels = batch[9].to(config.device)
    
    with torch.no_grad():
    # Forward pass, calculate logit predictions.  
    # token_type_ids is the same as "segment ids",  
    # which differentiates sentence 1 and 2 in 2-sentence tasks.  
    # values prior to applying an activation function like the softmax. 
        cc_input = (cc_input_ids, cc_input_mask, cc_input_types)
        td_input = (td_input_ids, td_input_mask, td_input_types)
        msg_input = (msg_input_ids, msg_input_mask, msg_input_types) 

        b_outputs = model(cc_input, td_input, msg_input)

    loss = F.cross_entropy(b_outputs, b_labels)
    # Accumulate the test loss.
    total_eval_loss += loss.item()
     
    # move labels to CPU 
    # preds_raw = b_outputs.data
    
    # torch tensor  
    lr = torch.nn.Softmax(dim=1)
    preds_prob = lr(b_outputs.data)
    # print("lr prob:", preds_prob, type(preds_prob), preds_prob.shape)
    # tensor to numpy  
    preds_prob = preds_prob.cpu().detach().numpy() 
    # print("lr prob:", preds_prob, type(preds_prob), preds_prob.shape)
    prob_result_lst.append( preds_prob ) 

    
    preds = torch.max(b_outputs.data, 1)[1].cpu().numpy()
    labels = b_labels.to('cpu').numpy()
    # print("preds:", type(preds), preds.shape)
    # print("preds:", preds)
    # print("labels:", type(labels), labels.shape)
    # print("labels:", labels)

    # Calculate the accuracy for this batch of test sentences, and
    total_eval_accuracy += flat_accuracy(preds, labels)
    # print("total eval acc:", total_eval_accuracy)
    # break

prob_result = numpy.vstack( prob_result_lst )
print("prob_result:", type(prob_result), prob_result.shape )

# output to csv   
numpy.savetxt("./data/train_data/train_prob.csv", prob_result, delimiter=",")

# Report the final accuracy for this testing run.
print(total_eval_accuracy)
print(len(train_dataloader))
avg_val_accuracy = total_eval_accuracy / len(train_dataloader)
print("  Accuracy: {0:.4f}".format(avg_val_accuracy))


# ========================================
#              Testing 
# ========================================

# After the completion of each training epoch, measure our performance on
# our test set. 
print("")
print("Running Testing...")

t0 = time.time()
# Put the model in evaluation mode--the dropout layers behave differently
# during evaluation.

# Tracking variables
total_eval_accuracy = 0
total_eval_loss = 0
nb_eval_steps = 0

prob_result_lst = []

# Evaluate data for one epoch 
for batch in test_dataloader:
    # Unpack this training batch from our dataloader 
    # As we unpack the batch, we will also copy each tensor to the GPU using 'to' method
    # 'batch' contains three pytorch tensors:
    # [0]: input ids
    # [1]: attention masks
    # [2]: labels
    cc_input_ids   = batch[0].to(config.device)
    cc_input_mask  = batch[1].to(config.device)
    cc_input_types = batch[2].to(config.device)
    
    td_input_ids   = batch[3].to(config.device)
    td_input_mask  = batch[4].to(config.device)
    td_input_types = batch[5].to(config.device)

    msg_input_ids   = batch[6].to(config.device)
    msg_input_mask  = batch[7].to(config.device)
    msg_input_types = batch[8].to(config.device)

    b_labels = batch[9].to(config.device)
    
    with torch.no_grad():
    # Forward pass, calculate logit predictions.  
    # token_type_ids is the same as "segment ids",  
    # which differentiates sentence 1 and 2 in 2-sentence tasks.  
    # values prior to applying an activation function like the softmax. 
        cc_input = (cc_input_ids, cc_input_mask, cc_input_types)
        td_input = (td_input_ids, td_input_mask, td_input_types)
        msg_input = (msg_input_ids, msg_input_mask, msg_input_types) 

        b_outputs = model(cc_input, td_input, msg_input)

    loss = F.cross_entropy(b_outputs, b_labels)
    # Accumulate the test loss.
    total_eval_loss += loss.item()
     
    # move labels to CPU 
    # preds_raw = b_outputs.data
    
    # torch tensor  
    lr = torch.nn.Softmax(dim=1)
    preds_prob = lr(b_outputs.data)
    # print("lr prob:", preds_prob, type(preds_prob), preds_prob.shape)
    # tensor to numpy  
    preds_prob = preds_prob.cpu().detach().numpy() 
    # print("lr prob:", preds_prob, type(preds_prob), preds_prob.shape)
    prob_result_lst.append( preds_prob ) 

    
    preds = torch.max(b_outputs.data, 1)[1].cpu().numpy()
    labels = b_labels.to('cpu').numpy()
    # print("preds:", type(preds), preds.shape)
    # print("preds:", preds)
    # print("labels:", type(labels), labels.shape)
    # print("labels:", labels)

    # Calculate the accuracy for this batch of test sentences, and
    total_eval_accuracy += flat_accuracy(preds, labels)
    # print("total eval acc:", total_eval_accuracy)
    # break

prob_result = numpy.vstack( prob_result_lst )
print("prob_result:", type(prob_result), prob_result.shape )

# output to csv   
numpy.savetxt("./data/test_data/test_prob.csv", prob_result, delimiter=",")

# Report the final accuracy for this testing run.
print(total_eval_accuracy)
print(len(test_dataloader))
avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

