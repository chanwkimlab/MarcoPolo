from torch.utils.data import Dataset



class CellDataset(Dataset):
    def __init__(self, y, x, s):
        self.y = y
        self.x = x
        self.s = s

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        item = {"Y": self.y[idx, :], "X": self.x[idx, :], "s": self.s[idx]}
        return item


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from scipy.io import mmread
    import numpy as np
    import pandas as pd
    Y_=mmread('../datasets/koh_extract/koh.data.counts.mm').toarray().astype(float).transpose()
    s_=pd.read_csv('../datasets/analysis/koh.size_factor_cluster.tsv',sep='\t',header=None)[0].values.astype(float)#.reshape(-1,1)
    X_=np.array([np.ones(Y_.shape[0])]).transpose()

    cell_dataset = CellDataset(Y_, X_, s_)
    cell_dataloader = DataLoader(dataset=cell_dataset, shuffle=False, batch_size=Y_.shape[0], num_workers=0)
