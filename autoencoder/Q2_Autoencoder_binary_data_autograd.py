import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt


p=0.2
N=1000
X1=bernoulli.rvs(p,size=N)
X1=np.array(X1).reshape(N,1)
X2=bernoulli.rvs(p,size=N)
X2=np.array(X2).reshape(N,1)
X3=bernoulli.rvs(p,size=N)
X3=np.array(X3).reshape(N,1)
#print("The shape of r is ",r.shape)

dataset=np.concatenate((X1,X2,X3), axis=1)
indices1 = np.random.permutation(dataset.shape[0])
training_idx, test_idx = indices1[:800], indices1[800:]
training_dataset, test_dataset = dataset[training_idx,:], dataset[test_idx,:]
label_train,label_test=dataset[training_idx,:], dataset[test_idx,:]
x_train, x_train_label, x_test, x_test_label = training_dataset,training_dataset,test_dataset,test_dataset
x_train = x_train.reshape(-1, x_train.shape[1]).astype('float32')
x_train_label = x_train
x_test = x_test.reshape(-1, x_test.shape[1]).astype('float32')
x_test_label = x_test
x_test.shape




from torch.utils.data import Dataset, DataLoader
class DataTrain(Dataset):
    def __init__(self):
        self.x=torch.from_numpy(x_train).type(torch.FloatTensor)
        self.len=self.x.shape[0]
    def __getitem__(self,index):      
        return self.x[index]
    def __len__(self):
        return self.len
class DataTest(Dataset):
    def __init__(self):
        self.x=torch.from_numpy(x_test).type(torch.FloatTensor)
        self.len=self.x.shape[0]
    def __getitem__(self,index):      
        return self.x[index]
    def __len__(self):
        return self.len



data_train=DataTrain()
data_test=DataTest()



trainloader=DataLoader(dataset=data_train,batch_size=1)
testloader=DataLoader(dataset=data_test,batch_size=1)




class AE(nn.Module):
    def __init__(self,D_in,D_H,D_out):
        super(AE,self).__init__()
        self.linear1=nn.Linear(D_in,D_H)
        self.linear2=nn.Linear(D_H,D_out)

        
    def forward(self,x):
        x=torch.sigmoid(self.linear1(x))  
        x=self.linear2(x)
        return x




input_dimenssion=3   #no of features in input
hidden_dimenssion = 2 # hidden layers
output_dimenssion=3# output layers




model=AE(input_dimenssion,hidden_dimenssion,output_dimenssion)




learning_rate=0.1
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
distance = nn.BCEWithLogitsLoss()




n_epochs=200
loss_train=[]
loss_test=[]
total_loss_train=[]
total_loss_test=[]
#n_epochs
for epoch in range(n_epochs):
    for x in trainloader:     
        optimizer.zero_grad()
        predict_train=model(x)
        loss=distance(predict_train,x) 
        loss.backward()
        optimizer.step()
        loss_train.append(loss.data)
    for x1 in testloader:
        predict_test=model(x1)
        losstest=distance(predict_test,x1)     
        loss_test.append(losstest.data)
    total_loss_train.append(np.mean(loss_train))
    total_loss_test.append(np.mean(loss_test))
    print('epoch {}, loss {}, losstest {}'.format(epoch, np.mean(loss_train),np.mean(loss_test)))



plt.title("Train and Test Loss each Epoch for binary data using Autograd")
plt.xlabel("Epochs")
plt.ylabel("Train and Test Loss for Autograd")
for i in range(198):
    plt.plot([i,i+2],total_loss_train[i:i+2],linestyle='-',linewidth=1,color='red')
    plt.plot([i,i+2],total_loss_test[i:i+2],linestyle='-',linewidth=1,color='blue')
plt.legend(["loss_train", "loss_test"], loc ="upper right")
plt.savefig("Q2_A.png")
plt.clf()