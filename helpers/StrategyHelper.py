from typing import List, Tuple
import pandas as pd


class StrategyHelper:
    def __init__(self,dataset,model)->None:
        self.dataset=dataset
        self.model=model
        
#    This is executed in query strategies
    def query(self)->None:
        pass
        
    def update(self,pos_index,neg_index=None)->None:
        self.dataset.labeled_index[pos_index]=True
        if neg_index:
            self.dataset.labeled_index[neg_index]=True
            
    def train(self)->None:
#        labeled_index, labeled_data_X, labeled_data_X 
        data_tuple=self.dataset.get_labeled_data()
        self.model.train(data_tuple)
        
    def predict(self,inp_data:Tuple[pd.Series,pd.Series])->List:
        X_test, y_test=inp_data
        out=self.model.predict(X_test)
        return out
    
#    def predict_prob(self,inp_data):
#        out_prob=self.model.predict_prob(inp_data)
#        return out_prob