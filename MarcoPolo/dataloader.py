#!/usr/bin/env python
# coding: utf-8

# In[6]:


from torch.utils.data import Dataset


# In[7]:


class Cell_Dataset(Dataset):
    def __init__(self,Y,X,s):
        self.Y=Y
        self.X=X
        self.s=s
        
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self,idx):
        item= {"Y":self.Y[idx,:],"X":self.X[idx,:],"s":self.s[idx]}
        return item  


# In[14]:


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from scipy.io import mmread
    import numpy as np
    import pandas as pd
    Y_=mmread('../datasets/koh_extract/koh.data.counts.mm').toarray().astype(float).transpose()
    s_=pd.read_csv('../datasets/analysis/koh.size_factor_cluster.tsv',sep='\t',header=None)[0].values.astype(float)#.reshape(-1,1)
    X_=np.array([np.ones(Y_.shape[0])]).transpose()    
    
    cell_dataset=Cell_Dataset(Y_,X_,s_)
    cell_dataloader=DataLoader(dataset=cell_dataset,shuffle=False,batch_size=Y_.shape[0],num_workers=0)    

