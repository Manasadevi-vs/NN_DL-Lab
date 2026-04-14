# LAB 8: Design and train a CNN on MNIST dataset.
from tensorflow.keras import *
import matplotlib.pyplot as plt
(xtr,ytr),(xte,yte)=datasets.mnist.load_data()
xtr,xte=xtr/255.0,xte/255.0
xtr=xtr[...,None]; xte=xte[...,None]
ytr=utils.to_categorical(ytr); yte=utils.to_categorical(yte)
m=models.Sequential([
layers.Conv2D(32,3,activation='relu',input_shape=(28,28,1)),
layers.MaxPooling2D(),
layers.Flatten(),
layers.Dense(10,activation='softmax')])
m.compile('adam','categorical_crossentropy',['accuracy'])
h=m.fit(xtr,ytr,epochs=3,validation_data=(xte,yte))
print(m.evaluate(xte,yte))
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(['train','test'])
plt.show()


# LAB 9: Design a neural network and demonstrate training using PyTorch.
import torch,torch.nn as nn,torchvision as tv
from torch.utils.data import DataLoader
tr=DataLoader(tv.datasets.FakeData(transform=tv.transforms.ToTensor()),batch_size=32)
m=nn.Sequential(nn.Flatten(),nn.Linear(3*224*224,10))
opt=torch.optim.Adam(m.parameters())
loss=nn.CrossEntropyLoss()
for x,y in tr:
    opt.zero_grad()
    l=loss(m(x),y)
    l.backward(); opt.step()
    break
print("Done")


# LAB 10: Apply optimization methods (Adagrad, RMSprop, Adam) on a neural network.
import torch,torch.nn as nn,torchvision as tv
from torch.utils.data import DataLoader
tr=DataLoader(tv.datasets.MNIST('.',download=True,
transform=tv.transforms.ToTensor()),batch_size=64)
m=nn.Sequential(nn.Flatten(),nn.Linear(784,10))
loss=nn.CrossEntropyLoss()
for opt in [torch.optim.Adagrad,torch.optim.RMSprop,torch.optim.Adam]:
    o=opt(m.parameters(),lr=0.01)
    for x,y in tr:
        o.zero_grad()
        l=loss(m(x),y)
        l.backward(); o.step()
        break
    print(opt.__name__)


# LAB 11: Implement and train a CNN architecture (LeNet) on MNIST dataset.
import torch,torch.nn as nn,torchvision as tv
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
tr=DataLoader(tv.datasets.MNIST('.',download=True,
transform=tv.transforms.ToTensor()),batch_size=64)
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.c=nn.Conv2d(1,6,5)
        s.f=nn.Linear(6*24*24,10)
    def forward(s,x):
        x=s.c(x)
        return s.f(x.view(x.size(0),-1))
m=Net(); opt=torch.optim.Adam(m.parameters())
loss=nn.CrossEntropyLoss()
losses=[]
for x,y in tr:
    opt.zero_grad()
    l=loss(m(x),y)
    l.backward(); opt.step()
    losses.append(l.item())
    break
print("CNN done")
plt.plot(losses)
plt.title("Loss")
plt.show()
