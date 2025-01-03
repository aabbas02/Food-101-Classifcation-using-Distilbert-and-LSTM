import sys
import pandas as pd
import numpy as np
import transformers
import os
import re
#import plotly.express as px
#import plotly.graph_objects as go
import torch
import torch.nn as nn
import sys

# %%
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


# %% [markdown]
# # PREPROCESSING 

# %% [markdown]
# ##  1) Custom Dataset Class
# #### Implement a map-style dataset from .csv files; needed because data files train_titles.csv and test_titles.csv cannot be directly passed to the dataloader

# %%
import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset

class customTextDataset(Dataset):
    def __init__(self, path, colnames, maxLen, tokenizer=None): 
        self.data =  pd.read_csv(path, names=colnames, header=None, sep = ',', index_col=False)
        self.tokenizer = tokenizer
        self.maxLen = maxLen

    """
    def __getitem__(self, idx):
        sentence = self.data.loc[idx].text
        #ids = self.tokenizer(self.tokenizer.bos_token + ' ' + sentence + ' ' + self.tokenizer.eos_token,  padding = 'max_length', truncation = True, max_length = self.maxLen)['input_ids']
        out = self.tokenizer(sentence, padding = 'max_length', truncation = True, max_length = self.maxLen)
        ids = out['input_ids']
        mask = torch.tensor(out['attention_mask'])
        ids = torch.tensor(ids)
        out = {'input_ids':ids, 'attention_mask':mask}
        # mask
        #padId = torch.unsqueeze(torch.tensor(self.tokenizer.encode(self.tokenizer.pad_token)[0]),axis=-1)
        #mask = (ids == padId)
        #mask = mask.repeat(self.maxLen,1)
        label = self.data.loc[idx].food
        return out, label
        #return torch.tensor(ids),label # torch.tensor(sentence), mask, label
    """
    def __getitem__(self, idx):
        sentence = self.data.loc[idx].text
        out = self.tokenizer(sentence, padding = 'max_length', truncation = True, max_length = self.maxLen)
        ids = out['input_ids']
        mask = torch.tensor(out['attention_mask'])
        ids = torch.tensor(ids)
        # label
        label = self.data.loc[idx].food
        return ids, mask, label
        #return torch.tensor(ids),label # torch.tensor(sentence), mask, label


    def __len__(self):
        return len(self.data)

    def getHead(self):
        print(self.data.head())



# %% [markdown]
# ## 2) DistilBert Tokenizer from Huggingface

# %%
from torch.utils.data import DataLoader
maxLen = 64
batchSize = 2048
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
vocabSize = len(tokenizer)
print(f"Vocabulary Size = {vocabSize}")

# %% [markdown]
# ## 3) Create Dataloaders from Custom Text Dataset

# %%
trainData = customTextDataset(path = 'train_titles.csv', colnames=['image_path', 'text', 'food'], tokenizer = tokenizer,maxLen=maxLen)
trainLoader = DataLoader(trainData,batch_size=batchSize,shuffle=True)
print(f'Number of train data points  = {trainData.__len__()}')
#trainData.getHead()
testData = customTextDataset(path = 'test_titles.csv', colnames=['image_path', 'text', 'food'], tokenizer = tokenizer,maxLen=maxLen)
testLoader = DataLoader(testData,batch_size=testData.__len__(),shuffle=True)
print(f"Number of test data points = {testData.__len__()}")

# %% [markdown]
# #### Print an example data point (tokens/tokenIds) and label

# %%
data = next(iter(trainLoader))
print(f"Encoded Text = {data[0][0]}, {data[0][0].type()}, {data[0][0].shape}")
print(f"Mask = {data[1][0]}, {data[1][0].type()}, {data[1][0].shape}")
print(f"Encoded Label = {data[2][0]}")
print(f"Decoded tokens from encoded ids: \n'{tokenizer.decode(data[0][0])}'")

labelsTrain_ = []
for data in trainLoader:
    labelsTrain_ = labelsTrain_ + (list(data[2]))
  
from collections import Counter
labelsDict = Counter(labelsTrain_)
keys = labelsDict.keys()
lblMap = {x:i for i,x in enumerate(keys)}
print(f"Number of classes = {len(list(keys))}") # This should be 101 

# %% [markdown]
# # DISTILBERT + FEEDFORWARD MODEL

from transformers import GPT2Config
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
numClasses = 101
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels = numClasses)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
lossVals = []
model = model.to(device)
#print(model)
numEpochs = 25
# %% [markdown]
# ## 1) Train Transformer + Feedforward Model

# %%
for epoch in range(numEpochs):  
    for data in trainLoader:
        ids, masks,labels_ = data 
        labels = torch.tensor([lblMap[x] for x in labels_]) 
        optimizer.zero_grad()
        ids = ids.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        outputs = model(input_ids = ids, attention_mask = masks,labels = labels)
        loss = outputs[0]
        loss.backward()
        lossVals.append(loss.detach().cpu().clone().numpy())
        optimizer.step()
    if epoch%1 == 0:
        print(f"Epoch = {epoch}. Loss = {loss}",flush=True)
print('Finished Training')

model.eval() # again no gradients needed - so set mode.eval()
correct_pred = 0
numTst = 0
with torch.no_grad():
    for data in testLoader:
        ids,masks,labels_ = data
        numTst = numTst + ids.shape[0]
        labels = []
        labels.append([lblMap[x] for x in labels_])
        labels = torch.tensor(labels[0])
        ids = ids.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        outputs = model(input_ids = ids, attention_mask = masks, labels = labels)
        _, predictions = torch.max(outputs[1], 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred += 1

accuracy = 100 * float(correct_pred)/ numTst
print(f"Classification Accuracy = {accuracy:.3f}%",flush=True)

import numpy as np
fig, ax = plt.subplots(nrows = 1, ncols=1)
ax.plot(range(len(lossVals)),lossVals)
ax.set_xlabel('Iterations',fontsize = 15)
ax.set_ylabel('Cross Entropy Loss', fontsize = 15)
ax.set_title('Classification Accuracy = {:.2f}%'.format(accuracy),fontsize = 15)
path = 'dim_{}_accry_{:.2f}len_{}_hidden_{}'.format(dModel,accuracy, maxLen,hidden_size)
#plt.savefig(path+'.pdf')

# %% [markdown]
# # SVM With LSTM 
# ### Instead of the 2 fully connected layers and 1 soft-max layer in the LSTM model above, we use multi-class SVM for classification below.

# %% [markdown]
# ## 1) Declare LSTM model and load trained weights.

# %%
class w2nModelSVM(torch.nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_size, nClasses):
        super(w2nModelSVM, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.FC0 = nn.FC0(input_size = embedding_dim, hidden_size = hidden_size,batch_first=True)

    def forward(self, x):
        x = self.word_embeddings(x)     # output dimensions is batch size = N x sequence length x feature size
        (x,_) = self.LSTM(x)        
        x = x[:, -1, :]                 # gives two dimensional output, not three dimensional output
        return x

modelSVM = w2nModelSVM(vocab_size = vocabSize, 
                 embedding_dim = dModel,
                 hidden_size=hidden_size, nClasses = numClasses
                )
modelSVM.word_embeddings.weight.data.copy_((model.word_embeddings.weight))
modelSVM.FC0.load_state_dict(model.FC0.state_dict())

# %% [markdown]
# ## 2) Get LSTM Embeddings for Train and Test Dataset

# %%
numTrn = trainData.__len__()
trnEmbdngs = np.zeros((numTrn,hidden_size))
trnLbls  = []
modelSVM.eval()
for i, data in enumerate(trainLoader):
    inputs,labels_ = data
    outputs = modelSVM(inputs)
    trnEmbdngs[i*batchSize: (i+1)*batchSize,:] = outputs.detach().clone().numpy()
    trnLbls = trnLbls + [lblMap[x] for x in labels_]

numTst = testData.__len__()
tstEmbdngs = np.zeros((numTst,hidden_size))
tstLbls = [] 
for i, data in enumerate(testLoader):
    inputs,labels_ = data
    labels =  []
    outputs = modelSVM(inputs)
    tstEmbdngs = outputs.detach().clone().numpy()
    tstLbls = tstLbls + [lblMap[x] for x in labels_]

# %% [markdown]
# ## 3) Accuracy using Linear SVM
# 

# %%
from sklearn import svm
clf = svm.SVC(decision_function_shape='ovo', kernel = 'linear')
clf.fit(trnEmbdngs, np.asarray(trnLbls))
TrnAccrcyLnr = clf.score(trnEmbdngs, np.asarray(trnLbls))
TstAccrcyLnr = clf.score(tstEmbdngs,np.asarray(tstLbls))
print(r'Train Accuracy of Linear SVM =', 100*TrnAccrcyLnr)
print(r'Test Accuracy of Linear SVM =', 100*TstAccrcyLnr)

# %% [markdown]
# ## 4) Accuracy using RBF Kernel SVM

# %%
clf = svm.SVC(decision_function_shape='ovo', kernel='rbf')
clf.fit(trnEmbdngs, trnLbls)
TrnAccrcyKrnl = clf.score(trnEmbdngs, trnLbls)
TstAccrcyKrnl = clf.score(tstEmbdngs,tstLbls)
print(r'Train Accuracy of Kernel SVM =', 100*TrnAccrcyKrnl)
print(r'Test Accuracy of Kernel SVM =', 100*TstAccrcyKrnl)


