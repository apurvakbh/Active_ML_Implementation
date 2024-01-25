#import argparse
import numpy as np
import torch
from helpers.DatasetHelper import DatasetHelper
from helpers.ModelHelper import Model
from query_strategies.RandomSampling import RandomSampling
from query_strategies.MarginSampling import MarginSampling
from query_strategies.MarginSamplingDropout import MarginSamplingDropout
from query_strategies.LeastConfidence import LeastConfidence

#Setting seed values to get deterministic results
RANDOM_SEED=7
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.enabled = False


def get_strategy(inp_strg:str):
    if inp_strg=="RandomSampling":
        return RandomSampling
    elif inp_strg=="MarginSampling":
        return MarginSampling
    elif inp_strg=="LeastConfidence":
        return LeastConfidence
    

if __name__ == "__main__":
    
    dataset=DatasetHelper.get_splitted_data("user_data/review_polarity/full_dataset.csv")
    dataset.initialize_labels(50)#change accordingly
    
    model=Model(dataset)
    
    strategy=get_strategy("LeastConfidence")(dataset,model)
    strategy.train()
    
    
    exit()
    
    preds = strategy.predict(dataset.get_test_data())
    
    for rd in range(1, 10):
        print(f"Round {rd}")

        query_idxs = strategy.query(5)

        # update labels
        strategy.update(query_idxs)
        strategy.train()

        # calculate accuracy
        preds = strategy.predict(dataset.get_test_data())
        print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds)}")