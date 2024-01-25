from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DatasetHelper:
    def __init__(self,X_train:pd.Series, X_test:pd.Series, y_train:pd.Series, y_test:pd.Series)->None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.labeled_index=np.zeros(len(self.X_train),dtype=bool)
        
    def get_dimensions(self)->Tuple[int,int]:
        return len(self.X_train[0]), len(self.y_train.unique())
    
    def initialize_labels(self,cnt:int)->None:
        random_index=np.arange(len(self.X_train))
        np.random.shuffle(random_index)
        self.labeled_index[random_index[:cnt]]=True
    
    def get_labeled_data(self)->Tuple[np.array,pd.Series,pd.Series]:
        labeled_index=np.arange(len(self.X_train))[self.labeled_index]
        return labeled_index, self.X_train[self.labeled_index], self.y_train[self.labeled_index]
    
    def get_unlabeled_data(self)->Tuple[np.array,pd.Series,pd.Series]:
        unlabeled_index=np.arange(len(self.X_train))[~self.labeled_index]
        return unlabeled_index, self.X_train[unlabeled_index], self.y_train[unlabeled_index]
    
    def get_train_data(self)->Tuple[np.array,pd.Series,pd.Series]:
        return self.labeled_index, self.X_train, self.y_train
    
    def get_test_data(self)->Tuple[pd.Series,pd.Series]:
        return self.X_test, self.y_test
    
    @staticmethod
    def __read_csv_data(file_loc:str)->pd.DataFrame:
        df_curr=pd.read_csv(file_loc)
        return df_curr
    
    @classmethod
    def get_splitted_data(cls,file_loc:str,inp_col:str="text",out_col:str="labels"):
        
        df=DatasetHelper.__read_csv_data(file_loc)
        X=df[inp_col]
        y=df[out_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

        return cls(X_train, X_test, y_train, y_test)
        