import torch
class TransformBase:
    '''
    VCBench.data.transforms.Base
    '''
    def __init__(self,obs_df):
        #assert obs_df
        #assert mode
        self.obs_df = obs_df

    def __call__(self, example):
        pass



