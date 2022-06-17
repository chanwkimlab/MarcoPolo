import datetime
import multiprocessing

import anndata as ad
import numpy as np
import pandas as pd
from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm

from MarcoPolo.regression.models import MarcoPoloModel
from MarcoPolo.regression.datasets import CellDataset

torch.set_default_dtype(torch.float64)


def fit_one_gene(model: nn.Module, optimizer: optim.Adamax, cell_dataloader: DataLoader, device: str,
                 EM_ITER_MAX: float, M_ITER_MAX: float, LL_diff_tolerance: float, Q_diff_tolerance: float,
                 verbose: bool = True):
    """
    Run EM trick algorithm.
    Args:
        model: Model to be trained
        optimizer: Optimizer to be used
        cell_dataloader: DataLoader for training
        device: device to use. If you want to use GPU set to 'cuda:0'. If you want to use CPU set to 'cpu'
        EM_ITER_MAX: maximum number of iterations of E-step of the EM trick algorithm
        M_ITER_MAX: maximum number of iterations of M-step of the EM trick algorithm
        LL_diff_tolerance: tolerance for the difference of log likelihood between two iterations of the EM trick algorithm
        Q_diff_tolerance: tolerance for the difference of Q between two iterations of the EM trick algorithm
        verbose:

    Returns:
        gamma_new: gamma after EM trick algorithm
        LL_new: log likelihood after EM trick algorithm
        Q_new: Q after EM trick algorithm
        em_idx_max: number of iterations of the EM trick algorithm
        m_idx_max: number of iterations of the M-step of the EM trick algorithm
    """
    if verbose:
        print('Start time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    with torch.no_grad():
        for batch_idx, batch in enumerate(cell_dataloader):
            batch_Y = batch['Y'].to(device)
            batch_X = batch['X'].to(device)
            batch_s = batch['s'].to(device)
        LL_old = model(batch_Y, batch_X, batch_s)
        Q_old = LL_old
    if verbose:
        print(LL_old)

    em_idx_max = 0
    m_idx_max = 0

    for em_idx in range(EM_ITER_MAX):  #
        LL_new = torch.zeros_like(LL_old)
        for batch_idx, batch in enumerate(cell_dataloader):
            # Usually batch size is the size of samples and only one batch is used. But if the dataset is too large, it is better to use multiple batches.
            batch_Y = batch['Y'].to(device)
            batch_X = batch['X'].to(device)
            batch_s = batch['s'].to(device)

            #############
            # M-step
            #############
            for m_idx in range(M_ITER_MAX):
                optimizer.zero_grad()
                Q_new = -model(batch_Y, batch_X, batch_s)
                Q_new.backward()
                optimizer.step()

                # Constraint
                model.delta_log.data = model.delta_log.data.clamp(min=model.delta_log_min)
                # model.NB_basis_a.data=model.NB_basis_a.data.clamp(min=0)

                if m_idx % 20 == 0:
                    Q_diff = (Q_old - Q_new) / torch.abs(Q_old)
                    Q_old = Q_new
                    if verbose:
                        print('M: {}, Q: {} Q_diff: {}'.format(m_idx, Q_new, Q_diff))
                    if m_idx > 0 and torch.abs(Q_diff) < (Q_diff_tolerance):
                        if verbose:
                            print('M break')
                        break
            m_idx_max = max(m_idx_max, m_idx)

            #############
            # Look at LL
            #############
            with torch.no_grad():
                LL_temp = -Q_new
                LL_new += LL_temp

        LL_diff = (LL_new - LL_old) / torch.abs(LL_old)
        LL_old = LL_new

        if verbose:
            print('EM: {}, LL: {} LL_diff: {}'.format(em_idx, LL_new, LL_diff))
        if LL_diff < LL_diff_tolerance:
            if verbose:
                print('EM break')
            break
    em_idx_max = max(em_idx_max, em_idx)

    if verbose:
        print('End time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    with torch.no_grad():
        gamma_new = model(batch_Y, batch_X, batch_s, to_return='gamma')

    return gamma_new, Q_new, LL_new, em_idx_max, m_idx_max


def fit_multiple_genes(Y: np.array, X: np.array, s: np.array,
                       num_cluster_list: List, learning_rate: float, fit_one_gene_parameters: dict, device: str,
                       start_gene_idx: Union[None, int] = None, end_gene_idx: Union[None, int] = None,
                       verbose: bool = False)-> dict:
    """
    Fit multiple genes.

    Args:
        Y (np.array): matrix of gene expression (cell, gene)
        X (np.array): matrix of cell covariates (cell, feature)
        s (np.array): vector of cell sizes (cell, )
        num_cluster_list (list): list of number of clusters to test
        learning_rate (float): learning rate of the optimizer
        fit_one_gene_parameters (dict): parameters for fit_one_gene
        start_gene_idx (Union[None, int]): start gene index. required for multithreading. If None, start from the first gene.
        end_gene_idx (Union[None, int]): end gene index. required for multithreading.  If None, end at the last gene.
        device (str): device to use. If you want to use GPU set to 'cuda:0'. If you want to use CPU set to 'cpu'
        verbose (bool): if True, print out the progress

    Returns:
        regression_result: a dictionary containing the regression results with the following fields: `gamma_list_cluster`, `delta_log_cluster`, `beta_cluster`, `result_cluster`.

    """

    Y_select = Y[:, start_gene_idx:end_gene_idx].copy()

    if len(multiprocessing.current_process()._identity) == 0 or multiprocessing.current_process()._identity[0] == 1:
        print(f'The numbers of clusters to test: {num_cluster_list}')
        print(f'Y: {Y.shape} X: {X.shape} s: {s.shape}')

    device = torch.device(device)

    gamma_cluster = {}

    Q_cluster = {}
    LL_cluster = {}
    em_idx_max_cluster = {}
    m_idx_max_cluster = {}

    delta_log_cluster = {}
    beta_cluster = {}

    for idx, num_cluster in enumerate(num_cluster_list):
        gamma_list = []

        Q_list = []
        LL_list = []
        em_idx_max_list = []
        m_idx_max_list = []

        delta_log_list = []
        beta_list = []

        if len(multiprocessing.current_process()._identity) == 0 or multiprocessing.current_process()._identity[0] == 1:
            print(f'({idx+1}) Fitting with {num_cluster} cluster(s)')
            pbar = tqdm(np.arange(Y_select.shape[1]), desc='Progress')
        else:
            pbar = np.arange(Y_select.shape[1])

        for iter_idx, exp_data_idx in enumerate(pbar):
            cell_dataset = CellDataset(Y_select[:, iter_idx:iter_idx + 1], X, s)

            cell_dataloader = DataLoader(dataset=cell_dataset, shuffle=False, batch_size=Y_select.shape[0],
                                         num_workers=0)

            if iter_idx == 0:
                model = MarcoPoloModel(Y=Y_select[:, iter_idx:iter_idx + 1], rho=np.ones((num_cluster, 1)),
                                       X_col=X.shape[1],
                                       delta_min=0).to(device)
            else:
                model.init_parameter_delta_min(0)
                model.init_paramter_Y(Y_select[:, iter_idx:iter_idx + 1])
            optimizer = optim.Adamax(model.parameters(), lr=learning_rate)  # ,betas=(0.92, 0.999))
            gamma, Q, LL, em_idx_max, m_idx_max = fit_one_gene(model=model, optimizer=optimizer,
                                                               cell_dataloader=cell_dataloader, device=device,
                                                               **fit_one_gene_parameters, verbose=verbose)

            gamma_list.append(gamma.cpu().numpy())

            Q_list.append(Q.detach().cpu().numpy())
            LL_list.append(LL.detach().cpu().numpy())
            em_idx_max_list.append(em_idx_max)
            m_idx_max_list.append(m_idx_max)

            delta_log_list.append(model.delta_log.detach().cpu().numpy())
            beta_list.append(model.beta.detach().cpu().numpy())

        gamma_cluster[num_cluster] = gamma_list

        delta_log_cluster[num_cluster] = delta_log_list

        beta_cluster[num_cluster] = beta_list

        Q_cluster[num_cluster] = Q_list
        LL_cluster[num_cluster] = LL_list
        em_idx_max_cluster[num_cluster] = em_idx_max_list
        m_idx_max_cluster[num_cluster] = m_idx_max_list

    result_cluster = {num_cluster: pd.DataFrame([Q_cluster[num_cluster],
                                                 LL_cluster[num_cluster],
                                                 em_idx_max_cluster[num_cluster],
                                                 m_idx_max_cluster[num_cluster]],
                                                index=['Q', 'LL', 'em_idx_max', 'm_idx_max']).T
                      for num_cluster in num_cluster_list}

    regression_result = {"gamma_list_cluster": gamma_cluster,
                         "delta_log_cluster": delta_log_cluster,
                         "beta_cluster": beta_cluster,
                         "result_cluster": result_cluster, }
    return regression_result


def run_regression(adata: ad.AnnData, size_factor_key: Union[str, None], covariates=None,
                   num_cluster_list=[1, 2], learning_rate=0.1,
                   EM_ITER_MAX=20, M_ITER_MAX=10000, LL_diff_tolerance=1e-4, Q_diff_tolerance=1e-4,
                   device: str='cuda:0', num_threads=1, verbose=False)->dict:
    """
    Run regression.

    Args:
        adata: anndata.AnnData containing scRNA-seq data. `.X` should be a matrix containing raw count data of shape (# cells, # genes).
        size_factor_key: key of the size factor stored in `adata.obs`. If not set, you can calculate size factor using `scanpy.pp.normalize_total` as follows. `norm_factor = sc.pp.normalize_total(adata, exclude_highly_expressed=True, max_fraction= 0.2, inplace=False)["norm_factor"]; adata.obs["size_factor"] = norm_factor/norm_factor.mean()` If None, no size factor is used.
        covariates: a covariate matrix of shape (# cells, # covariates). Default: None.
        num_cluster_list: a list of numbers of clusters to test. Default: [1, 2].
        learning_rate: learning rate of the Adamax optimizer. Default: 0.1.
        EM_ITER_MAX: maximum number of iterations of E-step of the EM trick algorithm. Default: 20.
        M_ITER_MAX: maximum number of iterations of M-step of the EM trick algorithm. Default: 10000.
        LL_diff_tolerance: tolerance of the difference of log-likelihood between two iterations of the EM trick algorithm. Default: 1e-4.
        Q_diff_tolerance: tolerance of the difference of Q between two iterations of the EM trick algorithm. Default: 1e-4.
        device: device to use. If you want to use GPU set to 'cuda:0'. If you want to use CPU set to 'cpu' Default: 'cuda:0'.
        verbose: if True, print the progress of the EM trick algorithm. Default: False.
        num_threads: number of threads to use. For each gene, MarcoPolo fits Poisson model to a matrix of (1, # cells). As the matrix is too small for us to fully utilize the power of GPU, it is good to use multiple threads at once. The best option depends on the number of cells and the GPU memory size. For 500 cells and 11GB, using 4 threads worked well. Default: 1.

    Returns:
        regression_result: a dictionary containing the regression results with the following fields: `gamma_list_cluster`, `delta_log_cluster`, `beta_cluster`, `result_cluster`.

    """
    if num_threads > 1 and device.startswith('cuda'):
        print(
            f"<INFO> Currently, you are using {num_threads} threads for regression. If you encounter any memory issues, try to set `num_threads` to 1.")

    expression_matrix = adata.X  # (cell, gene)
    num_cells = expression_matrix.shape[0]
    num_genes = expression_matrix.shape[1]

    if not type(expression_matrix) == np.ndarray:
        expression_matrix = expression_matrix.toarray().astype(float)
    else:
        expression_matrix = expression_matrix.astype(float)

    if size_factor_key is None:
        cell_size_factor = np.ones(expression_matrix.shape[0]).astype(float)
    else:
        cell_size_factor = adata.obs[size_factor_key].values.astype(float)

    if covariates is None:
        covariate_matrix = np.ones((expression_matrix.shape[0], 1)).astype(float)
    else:
        covariate_matrix = covariates.astype(float)

    fit_one_gene_parameters = {"EM_ITER_MAX": EM_ITER_MAX, "M_ITER_MAX": M_ITER_MAX,
                               "LL_diff_tolerance": LL_diff_tolerance, "Q_diff_tolerance": Q_diff_tolerance}

    if num_threads != 1:
        multiprocessing.set_start_method('spawn')
        pool = multiprocessing.Pool(processes=num_threads)

        gene_per_thread = expression_matrix.shape[1] // num_threads
        gene_thread_split = [(gene_per_thread * i, gene_per_thread * (i + 1)) for i in range(num_threads - 1)] + [
            (gene_per_thread * (num_threads - 1), expression_matrix.shape[1])]
        #gene_thread_split=gene_thread_split[::-1]

        multiprocessing.freeze_support()

        fit_result_thread = pool.starmap(fit_multiple_genes, [(expression_matrix,
                                                               covariate_matrix[:],
                                                               cell_size_factor[:],
                                                               num_cluster_list,
                                                               learning_rate,
                                                               fit_one_gene_parameters,
                                                               device,
                                                               start_gene_idx,
                                                               end_gene_idx,
                                                               verbose) for start_gene_idx, end_gene_idx in
                                                              gene_thread_split])

        pool.close()

        regression_result = {}

        for fit_result_thread in fit_result_thread[:]:
            for category, value_cluster in fit_result_thread.items():
                for num_cluster, value in value_cluster.items():
                    if isinstance(value, list):
                        regression_result[category][num_cluster] = regression_result.setdefault(category, {}).get(
                            num_cluster, []) + value
                    elif isinstance(value, pd.DataFrame):
                        regression_result[category][num_cluster] = regression_result.setdefault(category, {}).get(
                            num_cluster, []) + [value.reset_index()]
                    else:
                        raise ValueError("Unknown type of value: {}".format(type(value)))

        for category in regression_result.keys():
            for num_cluster in regression_result[category].keys():
                if isinstance(regression_result[category][num_cluster][0], pd.DataFrame):
                    regression_result[category][num_cluster] = pd.concat(
                        regression_result[category][num_cluster]).reset_index()
                assert len(regression_result[category][num_cluster])==num_genes, RuntimeError("Length of result is not equal to number of genes.")

    else:
        regression_result = fit_multiple_genes(Y=expression_matrix[:, :],
                                               X=covariate_matrix[:],
                                               s=cell_size_factor[:],
                                               num_cluster_list=num_cluster_list,
                                               learning_rate=learning_rate,
                                               fit_one_gene_parameters=fit_one_gene_parameters,
                                               device=device,
                                               verbose=verbose)

    return regression_result


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
