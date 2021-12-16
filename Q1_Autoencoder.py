import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

N=1000
mean1=[0,0,0]
cov1=[[1,0.8,0.8],[0.8,1,0.8],[0.8,0.8,1]]
dataset = np.random.multivariate_normal(mean1, cov1, N)
dataset = (dataset-np.amin(dataset))/(np.amax(dataset)-np.amin(dataset))

indices1 = np.random.permutation(dataset.shape[0])
training_idx, test_idx = indices1[:800], indices1[800:]
training_dataset, test_dataset = dataset[training_idx,:], dataset[test_idx,:]
label_train,label_test=dataset[training_idx,:], dataset[test_idx,:]

W11,W12,W13=0.01,0.03,0.05
W21,W22,W23=0.08,0.03,0.1
W0=0.01
W00=0.20

B11,B12=0.001,0.001
B21,B22=0.009,0.001
B31,B32=0.008,0.007
B0=0.001
B00=0.003
B000=0.004

#Initial Learning Rate
lr=0.1




def sigmoidF(X):
    return (1/(1+np.exp(-X)))
def sigmoidP(X):
    return (sigmoidF(X)*(1-sigmoidF(X)))



DW11=DW12=DW13=DW21=DW22=DW23=DW0=DW00=DB11=DB12=DB21=DB22=DB31=DB32=DB0=DB00=DB000=0.00




train_loss_array=[]
test_loss_array=[]




for epoch in range(200):
    y1=[]
    y2=[]
    y3=[]
    for i in range(training_dataset.shape[0]):
        X1=training_dataset[i][0]
        X2=training_dataset[i][1]
        X3=training_dataset[i][2]
        a1=(X1*W11)+(X2*W12)+(X3*W13)+W0
        Z1=sigmoidF(a1)
        a2=(X1*W21)+(X2*W22)+(X3*W23)+W00
        Z2=sigmoidF(a2)
        a3=(Z1*B11)+(Z2*B12)+B0
        a4=(Z1*B21)+(Z2*B22)+B00
        a5=(Z1*B31)+(Z2*B32)+B000
        Z3=sigmoidF(a3)
        Z4=sigmoidF(a4)
        Z5=sigmoidF(a5)
        y1.append(Z3)
        y2.append(Z4)
        y3.append(Z5)
        
        DW11=(-2*(X1-sigmoidF(a3))*sigmoidP(a3)*B11*sigmoidP(a1)*X1)+(-2*(X2-sigmoidF(a4))*sigmoidP(a4)*B21*sigmoidP(a1)*X1)+(-2*(X3-sigmoidF(a5))*sigmoidP(a5)*B31*sigmoidP(a1)*X1)
                
        DW12=(-2*(X1-sigmoidF(a3))*sigmoidP(a3)*B11*sigmoidP(a1)*X2)+(-2*(X2-sigmoidF(a4))*sigmoidP(a4)*B21*sigmoidP(a1)*X2)+(-2*(X3-sigmoidF(a5))*sigmoidP(a5)*B31*sigmoidP(a1)*X2)
                
        DW13=(-2*(X1-sigmoidF(a3))*sigmoidP(a3)*B11*sigmoidP(a1)*X3)+(-2*(X2-sigmoidF(a4))*sigmoidP(a4)*B21*sigmoidP(a1)*X3)+(-2*(X3-sigmoidF(a5))*sigmoidP(a5)*B31*sigmoidP(a1)*X3)
                
       
        DW21=(-2*(X1-sigmoidF(a3))*sigmoidP(a3)*B12*sigmoidP(a2)*X1)+(-2*(X2-sigmoidF(a4))*sigmoidP(a4)*B22*sigmoidP(a2)*X1)+ (-2*(X3-sigmoidF(a5))*sigmoidP(a5)*B32*sigmoidP(a2)*X1)
                
        DW22=(-2*(X1-sigmoidF(a3))*sigmoidP(a3)*B12*sigmoidP(a2)*X2)+(-2*(X2-sigmoidF(a4))*sigmoidP(a4)*B22*sigmoidP(a2)*X2)+(-2*(X3-sigmoidF(a5))*sigmoidP(a5)*B32*sigmoidP(a2)*X2)
                
        DW23=(-2*(X1-sigmoidF(a3))*sigmoidP(a3)*B12*sigmoidP(a2)*X3)+ (-2*(X2-sigmoidF(a4))*sigmoidP(a4)*B22*sigmoidP(a2)*X3)+(-2*(X3-sigmoidF(a5))*sigmoidP(a5)*B32*sigmoidP(a2)*X3)
        
        
        DW0=(-2*(X1-sigmoidF(a3))*sigmoidP(a3)*B11*sigmoidP(a1))+(-2*(X2-sigmoidF(a4))*sigmoidP(a4)*B21*sigmoidP(a1))+(-2*(X3-sigmoidF(a5))*sigmoidP(a5)*B31*sigmoidP(a1))       
        DW00=(-2*(X1-sigmoidF(a3))*sigmoidP(a3)*B11*sigmoidP(a2))+(-2*(X2-sigmoidF(a4))*sigmoidP(a4)*B21*sigmoidP(a2))+ (-2*(X3-sigmoidF(a5))*sigmoidP(a5)*B31*sigmoidP(a2))
                
        DB11=-2*(X1-sigmoidF(a3))*sigmoidP(a3)*Z1   
        DB12=-2*(X1-sigmoidF(a3))*sigmoidP(a3)*Z2
        
        DB21=-2*(X2-sigmoidF(a4))*sigmoidP(a4)*Z1   
        DB22=-2*(X2-sigmoidF(a4))*sigmoidP(a4)*Z2
        
        DB31=-2*(X3-sigmoidF(a5))*sigmoidP(a5)*Z1   
        DB32=-2*(X3-sigmoidF(a5))*sigmoidP(a5)*Z2
        
        DB0=-2*(X1-sigmoidF(a3))*sigmoidP(a3)
        DB00=-2*(X2-sigmoidF(a4))*sigmoidP(a4)
        DB000=-2*(X3-sigmoidF(a5))*sigmoidP(a5)
        
        W11=W11-(lr*DW11)
        W12=W12-(lr*DW12)
        W13=W13-(lr*DW13)
        W21=W21-(lr*DW21)
        W22=W22-(lr*DW22)
        W23=W23-(lr*DW23)
        W0=W0-(lr*DW0)
        W00=W00-(lr*DW00)
        
        B11=B11-(lr*DB11)
        B12=B12-(lr*DB12)
        B21=B21-(lr*DB21)
        B22=B22-(lr*DB22)
        B31=B31-(lr*DB31)
        B32=B32-(lr*DB32)
        
        B0=B0-(lr*DB0)
        B00=B00-(lr*DB00)
        B000=B000-(lr*DB000)
    diff1=np.square(training_dataset[:,0]-y1)
    diff2=np.square(training_dataset[:,1]-y2)
    diff3=np.square(training_dataset[:,2]-y3)
    diff=(diff1+diff2+diff3)
    difftrain=diff.reshape(800,1)
    msetrain=np.mean(difftrain)
    y11=[]
    y22=[]
    y33=[]
    for i in range(test_dataset.shape[0]):
        X1_test=test_dataset[i][0]
        X2_test=test_dataset[i][1]
        X3_test=test_dataset[i][2]
        a1=(X1_test*W11)+(X2_test*W12)+(X3_test*W13)+W0
        Z1=sigmoidF(a1)
        a2=(X1_test*W21)+(X2_test*W22)+(X3_test*W23)+W00
        Z2=sigmoidF(a2)
        a3=(Z1*B11)+(Z2*B12)+B0
        a4=(Z1*B21)+(Z2*B22)+B00
        a5=(Z1*B31)+(Z2*B32)+B000
        Z3=sigmoidF(a3)
        Z4=sigmoidF(a4)
        Z5=sigmoidF(a5)
        y11.append(Z3)
        y22.append(Z4)
        y33.append(Z5)
    diff1=np.square(test_dataset[:,0]-y11)
    diff2=np.square(test_dataset[:,1]-y22)
    diff3=np.square(test_dataset[:,2]-y33)
    diff=(diff1+diff2+diff3)
    difftest=diff.reshape(200,1)
    msetest=np.mean(difftest)
    print("Train Loss after Epoch" , epoch,"is : ",msetrain,"Test Loss after Epoch" , epoch,"is : ",msetest)
    train_loss_array.append(msetrain)
    test_loss_array.append(msetest)
    #lr=1/(epoch+1)




y11=[]
y22=[]
y33=[]
for i in range(test_dataset.shape[0]):
        X1=test_dataset[i][0]
        X2=test_dataset[i][1]
        X3=test_dataset[i][2]
        a1=(X1*W11)+(X2*W12)+(X3*W13)+W0
        Z1=sigmoidF(a1)
        a2=(X1*W21)+(X2*W22)+(X3*W23)+W00
        Z2=sigmoidF(a2)
        a3=(Z1*B11)+(Z2*B12)+B0
        a4=(Z1*B21)+(Z2*B22)+B00
        a5=(Z1*B31)+(Z2*B32)+B000
        Z3=sigmoidF(a3)
        Z4=sigmoidF(a4)
        Z5=sigmoidF(a5)
        y11.append(Z3)
        y22.append(Z4)
        y33.append(Z5)





diff1=(test_dataset[:,0]-y11)
diff2=(test_dataset[:,1]-y22)
diff3=(test_dataset[:,2]-y33)
diff=np.square(diff1+diff2+diff3)
diff=diff.reshape(200,1)




mse=np.mean(diff)
print("Loss on test set ",mse)




plt.title("Train and Test Loss each Epoch for guassian data without Autograd")
plt.xlabel("Epochs")
plt.ylabel("Train and Test Loss for without Autograd")
for i in range(198):
    plt.plot([i,i+2],train_loss_array[i:i+2],linestyle='-',linewidth=1,color='red')
    plt.plot([i,i+2],test_loss_array[i:i+2],linestyle='-',linewidth=1,color='blue')
plt.legend(["train_loss_array", "test_loss_array"], loc ="upper right")
plt.savefig("Q1_WA.png")
plt.clf()











