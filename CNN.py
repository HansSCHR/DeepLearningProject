### Master Degree Of Bioinformatics ###
### SCHRIEKE Hans ###
### Deep Learning Project ###
### 17/02/19


# coding: utf-8


import sys
import os


### READ FOLDER ###

def read_folder(path):
    files = os.listdir(path)
    for name in files:
        if name.find(" ")!= -1:
            os.rename(path+'/' + name, path+ '/' +name.replace(' ', '_'))


#Define dataset path
path_train="fruits-360/Training"
path_test="fruits-360/Test"

#Read the folder
read_folder(path_train)
read_folder(path_test)



### DATASET  ###

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms ,datasets

#Insert dataset
train_dataset= datasets.ImageFolder(path_train, transform = transforms.ToTensor())
train_loader= DataLoader(train_dataset,batch_size=4,shuffle= True)

test_dataset= datasets.ImageFolder(path_train, transform = transforms.ToTensor())
test_loader= DataLoader(test_dataset,batch_size=4,shuffle= True)



### CONVOLUTIONAL NEURAL NETWORK ###

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


class Net(nn.Module):
#"7.Definethelayersinthenetwork"
    def __init__(self):
        super(Net,self).__init__()

        #1input imagechannel,6outputchannels,5x5squareconvolutionkernel
        self.conv1=nn.Conv2d(3,70,5) #Conv 1
        self.conv2=nn.Conv2d(70,140,5) #Conv2 
	
        #anaffineoperation:y=Wx+b
        self.fc1=nn.Linear(140*22*22,300)#(sizeofinput,sizeofoutput)
        self.fc2=nn.Linear(300,200) 
        self.fc3=nn.Linear(200,83)
		
		#"Implementtheforwardcomputationofthenetwork"

    def forward(self,x):
        #Maxpoolingovera(2,2)window
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        #Ifthesizeisasquareyoucanonlyspecifyasinglenumber
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(-1,self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        #alldimensionsexceptthebatchdimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features


#Check the machine have GPU 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )
print (device)
net=Net().to(device)
print (net)
criterion=nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



### TRAINING ### 

epoch = 5 #Number of turns
print ("test")
for epoch in range (epoch):
    running_loss=0.0
    for i, data in enumerate(train_loader,0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs=net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i%2000==1999: #print every 2000 mini batches
            print ('[%d,%5d]loss:%.3f'%
                (epoch+1,i+1,running_loss/2000))
            running_loss=0.0
print ("Finished Training!!!")




import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img/2 + 0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    


### TEST ###

testiter = iter(test_loader)
images, labels =  testiter.next()
print (labels)

outputs = net(images)
_,predicted = torch.max(outputs,1)
print (predicted)



### RESULT ###

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs=net(images)
        _,predicted= torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
print ('Accuracy of the network on the 10000 test images %d %%' %(100*correct/total))





    
    





