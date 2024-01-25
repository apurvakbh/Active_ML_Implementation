from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import scipy
import numpy as np

from sklearn.metrics import accuracy_score

class NNBaseModel(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.tokenizer=CountVectorizer()
        
    def make_model(self,inp_shape, out_shape)->None:
        p=0.5
        self.flatten=nn.Flatten()
        self.model_full=nn.Sequential(
            nn.Linear(inp_shape,256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p),
            nn.Linear(256,32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(p),
            nn.Linear(32,8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Dropout(p),
            nn.Linear(8,out_shape)
        )
        
        self.last_layer=nn.Softmax(dim=1)
        
    def forward(self,x):
        x=self.model_full(x)
        x=self.last_layer(x).argmax(1)
        return x
    
    def __fit_tokenizer(self,inp_X:List[str])->None:
        self.tokenizer.fit(inp_X)
    
    def __tokenize(self,inp_X:List[str])->None:
        return self.tokenizer.transform(inp_X)
    
    def get_train_data_ready(self,train_data:Tuple[np.array,pd.Series,pd.Series])->Tuple[torch.tensor,torch.tensor]:
        X_train=train_data[1]
        y_train=train_data[2]
        
        self.__fit_tokenizer(X_train)
        
        X_train_tokenized=self.__tokenize(X_train)
        X=torch.tensor(scipy.sparse.csr_matrix.todense(X_train_tokenized), dtype=torch.float32)
        y=torch.tensor(y_train.values, dtype=torch.float32)
        
        return X, y
    
    def get_test_data_ready(self,inp_data:List[str])->torch.tensor:
        inp_data_tokenized=self.__tokenize(inp_data)
        inp_tensor=torch.tensor(scipy.sparse.csr_matrix.todense(inp_data_tokenized), dtype=torch.float32)        
        return inp_tensor
        

class Model:
    def __init__(self,dataset):
        self.dataset=dataset
        self.model=NNBaseModel()
        
    def train(self,data_tuple:Tuple[np.array,pd.Series,pd.Series],n_epochs:int=10)->None:     
        
        X, y=self.model.get_train_data_ready(data_tuple)
        
        self.model.train()
        
        loss_fn=nn.BCELoss()
        inp_shape, out_shape=self.dataset.get_dimensions()
        self.model.make_model(len(X[0]), out_shape)
        
        optimizer=optim.Adam(self.model.parameters(),lr=0.001)
        
        
        for epoch in range(n_epochs):
            y_pred = self.model(X)
            y_pred=y_pred.type("torch.FloatTensor")
            loss=loss_fn(y_pred,y)
            loss = Variable(loss, requires_grad = True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            if epoch%2==0:
                print(f'Finished epoch {epoch}, latest loss {loss}')
        
    def predict(self,inp_data:List[str])->List:
                
        self.model.eval()
        with torch.no_grad():
            model_input=self.model.get_test_data_ready(inp_data)
            out=self.model(model_input)
        
        out=out.tolist()
        return out
    
    

#class ModelTraining(BaseModel):
#    def __init__(self,inp_shape,out_shape):
#        _=1
#    
#    @staticmethod
#    def train_model(data_tuple:Tuple[pd.Series],n_epochs:int=40,batch_size:int=32)->BaseModel:
#        X_train, X_test, y_train, y_test=data_tuple
#        
#        vectorizer = CountVectorizer()
#        X_train = vectorizer.fit_transform(X_train)
#        X_test = vectorizer.transform(X_test)
#        
#        X=torch.tensor(scipy.sparse.csr_matrix.todense(X_train), dtype=torch.float32)#.data
#        y=torch.tensor(y_train.values, dtype=torch.float32)#.reshape(-1, 1)
#        
#        model=BaseModel(X.shape[1],len(y_train.unique()))
#        model.train()
#        
#        loss_fn=nn.BCELoss()
#        optimizer=optim.Adam(model.parameters(),lr=0.01)
#        
#        for epoch in range(n_epochs):
#            y_pred = model(X)
#            y_pred=y_pred.type("torch.FloatTensor")
#            loss=loss_fn(y_pred,y)
#            loss = Variable(loss, requires_grad = True)
#            optimizer.zero_grad()
#            loss.backward()
#            optimizer.step()            
#            if epoch%2==0:
#                print(f'Finished epoch {epoch}, latest loss {loss}')
#        
#        model.eval()
#        with torch.no_grad():
#            X_test=torch.tensor(scipy.sparse.csr_matrix.todense(X_test), dtype=torch.float32)
#            y_pred=model(X_test)
#        
#        y_pred=y_pred.tolist()
#        acc=accuracy_score(y_test, y_pred)
#        print("Accuray is ",acc)
#        
#        return model