import datetime
import sys
import multiprocessing

import anndata as ad
import numpy as np
import pandas as pd
from typing import Union, List, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm

from MarcoPolo.regression.models import MarcoPoloModel
from MarcoPolo.regression.datasets import CellDataset

torch.set_default_dtype(torch.float64)

def fit_one_gene(model, optimizer, cell_dataloader, device, EM_ITER_MAX, M_ITER_MAX, LL_diff_tolerance, Q_diff_tolerance, verbose=True):
    """
    Run EM trick algorithm with EM_ITER_MAX iterations.
    Args:
        model:
        optimizer:
        cell_dataloader:
        device:
        EM_ITER_MAX:
        M_ITER_MAX:
        LL_diff_tolerance:
        Q_diff_tolerance:
        verbose:

    Returns:

    """
    if verbose:
        print('Start time:',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    with torch.no_grad():
        for batch_idx,batch in enumerate(cell_dataloader):
            batch_Y=batch['Y'].to(device)
            batch_X=batch['X'].to(device)
            batch_s=batch['s'].to(device)
        LL_old=model(batch_Y,batch_X,batch_s)
        Q_old=LL_old
    if verbose:
        print(LL_old)

    em_idx_max=0
    m_idx_max=0

    for em_idx in range(EM_ITER_MAX):#
        LL_new=torch.zeros_like(LL_old)
        for batch_idx,batch in enumerate(cell_dataloader):
            # Usually batch size is the size of samples and only one batch is used. But if the dataset is too large, it is better to use multiple batches.
            batch_Y=batch['Y'].to(device)
            batch_X=batch['X'].to(device)
            batch_s=batch['s'].to(device)

            #############
            #M-step
            #############
            for m_idx in range(M_ITER_MAX):
                optimizer.zero_grad()
                Q_new=-model(batch_Y,batch_X,batch_s)
                Q_new.backward()
                optimizer.step()

                #Constraint
                model.delta_log.data=model.delta_log.data.clamp(min=model.delta_log_min)
                #model.NB_basis_a.data=model.NB_basis_a.data.clamp(min=0)

                if m_idx%20==0:
                    Q_diff=(Q_old-Q_new)/torch.abs(Q_old)
                    Q_old=Q_new
                    if verbose:
                        print('M: {}, Q: {} Q_diff: {}'.format(m_idx,Q_new,Q_diff))
                    if m_idx>0 and torch.abs(Q_diff)<(Q_diff_tolerance):
                        if verbose:
                            print('M break')
                        break
            m_idx_max=max(m_idx_max,m_idx)

            #############
            #Look at LL
            #############
            with torch.no_grad():
                LL_temp=-Q_new
                LL_new+=LL_temp

        LL_diff=(LL_new-LL_old)/torch.abs(LL_old)
        LL_old=LL_new

        if verbose:
            print('EM: {}, LL: {} LL_diff: {}'.format(em_idx,LL_new,LL_diff))
        if LL_diff<LL_diff_tolerance:
            if verbose:
                print('EM break')
            break
    em_idx_max=max(em_idx_max,em_idx)

    if verbose:
        print('End time:',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    with torch.no_grad():
        gamma_new=model(batch_Y,batch_X,batch_s,to_return='gamma')

    return gamma_new, Q_new, LL_new, em_idx_max, m_idx_max


def fit_multiple_genes(Y: np.array, X: np.array, s: np.array,
                       num_cluster_list: List, learning_rate: float, fit_one_gene_parameters: dict, device: str, start_gene_idx=None, end_gene_idx=None, verbose: bool=False):
    """

    Args:
        Y (np.array): (cell, gene)
        X (np.array): (cell, batch)
        s (np.array): (cell, )
        num_cluster_list (list):
        learning_rate (float):
        fit_one_gene_parameters (dict):
        device (str):
        verbose (bool):

    Returns:

    """

    #return {"a":[4,5,6], "b":[1,2,3]}

    Y_select= Y[:, start_gene_idx:end_gene_idx]
    #print(type(multiprocessing.current_process()._identity[0]), multiprocessing.current_process()._identity, multiprocessing.current_process()._identity[0]==0)
    if len(multiprocessing.current_process()._identity) == 0 or multiprocessing.current_process()._identity[0] == 1:
        print('num_cluster_list', num_cluster_list)
        print('Y: {} X: {} s: {}'.format(Y_select.shape, X.shape, s.shape))
    
    device=torch.device(device)
    
    gamma_cluster={}
    
    Q_cluster={}
    LL_cluster={}
    em_idx_max_cluster={}
    m_idx_max_cluster={}
    
    delta_log_cluster={}
    beta_cluster={}
    
    for num_cluster in num_cluster_list:
        gamma_list=[]
        
        Q_list=[]
        LL_list=[]
        em_idx_max_list=[]
        m_idx_max_list=[]
        
        delta_log_list=[]
        beta_list=[]

        if len(multiprocessing.current_process()._identity)==0 or multiprocessing.current_process()._identity[0]==1:
            print(f'Fitting with {num_cluster} cluster{num_cluster > 1 and "s" or ""}')
            tbar=tqdm(np.arange(Y_select.shape[1]), desc='Completed genes:')
        else:
            tbar=np.arange(Y_select.shape[1])

        for iter_idx, exp_data_idx in enumerate(tbar):
            cell_dataset = CellDataset(Y_select[:, iter_idx:iter_idx + 1], X, s)
            
            cell_dataloader=DataLoader(dataset=cell_dataset, shuffle=False, batch_size=Y_select.shape[0], num_workers=0)

            if iter_idx==0:
                model = MarcoPoloModel(Y=Y_select[:, iter_idx:iter_idx + 1], rho=np.ones((num_cluster, 1)), X_col=X.shape[1],
                                       delta_min=0).to(device)
            else:
                model.init_parameter_delta_min(0)
                model.init_paramter_Y(Y_select[:, iter_idx:iter_idx + 1])
            optimizer = optim.Adamax(model.parameters(),lr=learning_rate)#,betas=(0.92, 0.999))
            gamma_new, Q_new, LL_new, em_idx_max, m_idx_max=fit_one_gene(model=model, optimizer=optimizer, cell_dataloader=cell_dataloader, device=device,
                                                                         **fit_one_gene_parameters, verbose=verbose)
            
            gamma_list.append(gamma_new.cpu().numpy())
            
            Q_list.append(Q_new.detach().cpu().numpy())
            LL_list.append(LL_new.detach().cpu().numpy())
            em_idx_max_list.append(em_idx_max)
            m_idx_max_list.append(m_idx_max)
            
            delta_log_list.append(model.delta_log.detach().cpu().numpy())
            beta_list.append(model.beta.detach().cpu().numpy())

        gamma_cluster[num_cluster] = gamma_list

        delta_log_cluster[num_cluster] = delta_log_list

        beta_cluster[num_cluster] = beta_list

        Q_cluster[num_cluster]=Q_list
        LL_cluster[num_cluster]=LL_list
        em_idx_max_cluster[num_cluster]=em_idx_max_list
        m_idx_max_cluster[num_cluster]=m_idx_max_list

    result_cluster={num_cluster: pd.DataFrame([Q_cluster[num_cluster],
                                               LL_cluster[num_cluster],
                                               em_idx_max_cluster[num_cluster],
                                               m_idx_max_cluster[num_cluster]], index=['Q','LL','em_idx_max','m_idx_max']).T
                    for num_cluster in num_cluster_list}

    
    return {"gamma_list_cluster": gamma_cluster,
            "delta_log_cluster": delta_log_cluster,
            "beta_cluster": beta_cluster,
            "result_cluster": result_cluster,}


def run_regression(adata: ad.AnnData, size_factor_key: Union[str, None], covariates=None,
                   num_cluster_list=[1, 2], learning_rate=0.1,
                   EM_ITER_MAX=20, M_ITER_MAX=10000, LL_diff_tolerance=1e-4, Q_diff_tolerance=1e-4,
                   device='cuda:{}'.format(0), num_threads=1, verbose=False):
    """

    Args:
        adata:
        size_factor:
        output_path:
        covariates:
        num_cluster_list:
        learning_rate:
        EM_ITER_MAX:
        M_ITER_MAX:
        LL_diff_tolerance:
        Q_diff_tolerance:
        device:
        verbose:
        num_threads:

    Returns:

    """
    if num_threads > 1 and device.startswith('cuda'):
        print(f"<INFO> Currently, {num_threads} threads are being used for regression. If you encounter any memory issues, try to set num_threads to 1.")

    expression_matrix=adata.X # (cell, gene)
    if not type(expression_matrix)==np.ndarray:
        expression_matrix=expression_matrix.toarray().astype(float)
    else:
        expression_matrix=expression_matrix.astype(float)

    if size_factor_key is None:
        cell_size_factor = np.ones(expression_matrix.shape[0]).astype(float)
    else:
        cell_size_factor = adata.obs[size_factor_key].values.astype(float)

    if covariates is None:
        covariate_matrix=np.ones((expression_matrix.shape[0], 1)).astype(float)
    else:
        covariate_matrix=covariates.astype(float)

    fit_one_gene_parameters = {"EM_ITER_MAX": EM_ITER_MAX, "M_ITER_MAX": M_ITER_MAX, "LL_diff_tolerance": LL_diff_tolerance, "Q_diff_tolerance": Q_diff_tolerance}


    if num_threads!=1:
        pool=multiprocessing.Pool(processes=num_threads)

        gene_per_thread= expression_matrix.shape[1] // num_threads
        gene_thread_split= [(gene_per_thread*i, gene_per_thread*(i+1)) for i in range(num_threads - 1)] + [(gene_per_thread * (num_threads - 1), expression_matrix.shape[1])]


        # fit_result_thread=pool.starmap(fit_multiple_genes, tqdm.tqdm([(expression_matrix[:, start_gene_idx:end_gene_idx],
        #                                                     covariate_matrix[:],
        #                                                     cell_size_factor[:],
        #                                                     num_cluster_list,
        #                                                     learning_rate,
        #                                                     fit_one_gene_parameters,
        #                                                     device,
        #                                                     verbose) for start_gene_idx, end_gene_idx in gene_thread_split]))
        multiprocessing.freeze_support()
        fit_result_thread=pool.starmap(fit_multiple_genes, [(expression_matrix,
                                                                       covariate_matrix[:],
                                                                       cell_size_factor[:],
                                                                       num_cluster_list,
                                                                       learning_rate,
                                                                       fit_one_gene_parameters,
                                                                       device,
                                                                       start_gene_idx,
                                                                       end_gene_idx,
                                                                       verbose) for start_gene_idx, end_gene_idx in gene_thread_split])

        pool.close()

        fit_result={}

        for fit_result_thread in fit_result_thread[:]:
            for category, value_cluster in fit_result_thread.items():
                for num_cluster, value in value_cluster.items():
                    if isinstance(value, list):
                        fit_result[category][num_cluster]=fit_result.setdefault(category, {}).get(num_cluster, []) + value
                    elif isinstance(value, pd.DataFrame):
                        fit_result[category][num_cluster] = fit_result.setdefault(category, {}).get(num_cluster, []) + [value.reset_index()]
                    else:
                        raise ValueError("Unknown type of value: {}".format(type(value)))

        for category in fit_result.keys():
            for num_cluster in fit_result[category].keys():
                if isinstance(fit_result[category][num_cluster][0], pd.DataFrame):
                    fit_result[category][num_cluster]=pd.concat(fit_result[category][num_cluster]).reset_index()

    else:
        fit_result=fit_multiple_genes(Y=expression_matrix[:, :],
                                      X=covariate_matrix[:],
                                      s=cell_size_factor[:],
                                      num_cluster_list=num_cluster_list,
                                      learning_rate=learning_rate,
                                      fit_one_gene_parameters=fit_one_gene_parameters,
                                      device=device,
                                      verbose=verbose)

    return fit_result


if __name__ == '__main__':
    data_path = "/homes/gws/chanwkim/MarcoPolo/notebooks/example/hESC.h5ad"
    adata = ad.read(data_path)
    run_regression(adata=adata[:, :10], size_factor_key="size_factor", num_threads=3, device="cuda:2")


    # from scipy.io import mmread
    # import numpy as np
    # import pandas as pd
    # Y_=mmread('../datasets/koh_extract/koh.data.counts.mm').toarray().astype(float).transpose()
    # s_=pd.read_csv('../datasets/analysis/koh.size_factor_cluster.tsv',sep='\t',header=None)[0].values.astype(float)#.reshape(-1,1)
    # X_=np.array([np.ones(Y_.shape[0])]).transpose()

    device = torch.device('cuda:2')

    Y_ = np.ones((446, 4898))
    X_ = np.ones((446, 1))
    s_ = np.ones((446))

    fit_multiple_genes(Y_select=Y_, X=X_, s=s_, num_cluster_list=[1, 2, 3], LR=0.1, EM_ITER_MAX=20, M_ITER_MAX=10000,
                       LL_diff_tolerance=1e-4, Q_diff_tolerance=1e-4, device=device, verbose=True)
