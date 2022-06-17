import numpy as np
import pandas as pd
import anndata as ad

import warnings

from sklearn import preprocessing
from sklearn.decomposition import PCA

import MarcoPolo.utils

pd.options.mode.chained_assignment = None

def find_markers(adata: ad.AnnData, regression_result: dict, mode: float=2, voting_thres: float=0.7, PCA_norm_thres:float=10, num_PC:int=2, log_fold_change_thres:float=0.6,
                 oncell_size_min_count:float=10, oncell_size_max_proportion:float=70)->pd.DataFrame:
    """
    find markers from the regression result
    Args:
        adata: anndata.AnnData containing scRNA-seq data. `.X` should be a matrix containing raw count data of shape (# cells, # genes).
        regression_result: dict containing regression results. Return value of `run_regression` function.
        mode: the number of groups to be used for marker selection. Default: 2.
        voting_thres: the threshold for voting. should be between 0 and 1. Default: 0.7.
        PCA_norm_thres: the threshold for PCA normalization. Default: 10.
        num_PC: the number of PCs to be used for marker selection. should be between 1 and 50 Default: 2.
        log_fold_change_thres: the threshold for log fold change. Default: 0.6.
        oncell_size_min_count: the minimum number of cells in on-cell group. Default: 10.
        oncell_size_max_proportion: the maximum proportion of cells in on-cell group. Default: 70.

    Returns:
        gene_scores: a pandas.DataFrame containing the following columns: 'MarcoPolo_rank', 'bimodality_score', 'voting_score', 'proximity_score', etc.

    """
    expression_matrix = adata.X.copy()
    num_cells=expression_matrix.shape[0]
    num_genes=expression_matrix.shape[1]

    ########################
    # Assign cells to on-cells and off-cells
    ########################
    print("Assign cells to on-cells and off-cells...")
    gamma_list = regression_result["gamma_list_cluster"][mode]
    gamma_argmax_list = MarcoPolo.utils.gamma_list_expression_matrix_to_gamma_argmax_list(gamma_list, expression_matrix)

    ########################
    # Calculate log fold change
    ########################
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        log_fold_change = np.log10(np.array([np.mean(expression_matrix[gamma_argmax_list[i] == 0, i]) for i in
                                 range(num_genes)]) / np.array(
            [np.mean(expression_matrix[gamma_argmax_list[i] != 0, i]) for i in range(num_genes)]))

    ########################
    # Calculate voting score
    ########################
    print("Calculating voting score...")
    oncell_size_list = np.sum(gamma_argmax_list == 0, axis=1)
    oncell_size_cliplist = MarcoPolo.utils.gamma_argmax_list_to_oncell_size_list_list(gamma_argmax_list)
    intersection_list = MarcoPolo.utils.gamma_argmax_list_to_intersection_list(gamma_argmax_list)
    #intersectioncount_prop=((intersection_list/oncell_size_cliplist))
    #intersectioncount_prop_top10=[np.arange(0,len(i))[i>=sorted(i)[-10]][:10] for i in intersectioncount_prop]
    intersectioncount_threshold = ((intersection_list / oncell_size_cliplist) > voting_thres)
    voting_score = np.sum(intersectioncount_threshold, axis=1)

    ########################
    # Calculate proximity score
    ########################
    print("Calculating proximity score...")
    expression_matrix_norm = np.log1p(10000 * expression_matrix / expression_matrix.sum(axis=1, keepdims=True))
    #expression_matrix_norm_scale = preprocessing.scale(expression_matrix_norm, axis=0, with_mean=True, with_std=True, copy=True)
    expression_matrix_norm_scale=(expression_matrix_norm-expression_matrix_norm.mean(axis=0, keepdims=True))/expression_matrix_norm.std(axis=0, keepdims=True)
    expression_matrix_norm_scale[expression_matrix_norm_scale > PCA_norm_thres] = PCA_norm_thres

    pca = PCA(n_components=50)
    pca.fit(expression_matrix_norm_scale)
    expression_matrix_norm_scale_pc = pca.transform(expression_matrix_norm_scale)

    proximity_score = np.array(
        [expression_matrix_norm_scale_pc[gamma_argmax_list[i] == 0, :num_PC].std(axis=0).mean() for i in
         range(num_genes)])

    ########################
    # Calculate bimodality score
    ########################
    print("Calculating bimodality score...")
    QQratio = regression_result["result_cluster"][1]['Q'] / regression_result["result_cluster"][mode]['Q']
    mean_all = np.array([np.mean(expression_matrix[:, i]) for i in range(num_genes)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_on = np.array(
            [np.mean(expression_matrix[gamma_argmax_list[i] == 0, i]) for i in range(num_genes)])
    MS = mean_on - mean_all

    ########################
    # Final step of obtaining MarcoPolo score
    ########################
    print("Calculating MarcoPolo score...")
    gene_scores = pd.DataFrame([QQratio.values,
                                voting_score,
                                proximity_score,
                               log_fold_change,
                               MS,
                               oncell_size_list,],
                               index=['QQratio',
                                     'voting_score',
                                     'proximity_score',
                                     'log_fold_change',
                                     'MS',
                                     'oncell_size']).T

    gene_scores['QQratio_rank'] = \
        pd.Series(np.arange(num_genes), index=gene_scores['QQratio'].sort_values(ascending=False).index).loc[
            gene_scores.index]

    gene_scores['voting_score_rank'] = \
        pd.Series(np.arange(num_genes),
                  index=gene_scores['voting_score'].sort_values(ascending=False).index).loc[
            gene_scores.index]
    gene_scores['voting_score_rank'][gene_scores['voting_score'] == 0] = 499999
    gene_scores['voting_score_rank'][gene_scores['voting_score'] == 1] = 999999

    gene_scores['proximity_score_rank'] = \
        pd.Series(np.arange(num_genes), index=gene_scores['proximity_score'].sort_values(ascending=True).index).loc[
            gene_scores.index]

    gene_scores['log_fold_change_rank'] = \
        pd.Series(np.arange(num_genes), index=gene_scores['log_fold_change'].sort_values(ascending=False).index).loc[
            gene_scores.index]

    gene_scores['MS_rank'] = \
        pd.Series(np.arange(num_genes), index=gene_scores['MS'].sort_values(ascending=False).index).loc[
            gene_scores.index]

    gene_scores['oncell_size_rank'] = \
        pd.Series(np.arange(num_genes), index=gene_scores['oncell_size'].sort_values(ascending=False).index).loc[
            gene_scores.index]

    # Exclude outliers genes from ranking.
    gene_scores['voting_score_rank'][~(
            (gene_scores['log_fold_change'] > log_fold_change_thres) &
            (gene_scores['oncell_size'] > int(oncell_size_min_count)) &
            (gene_scores['oncell_size'] < int(oncell_size_max_proportion / 100 * num_cells))
    )] = len(gene_scores)

    gene_scores['bimodality_score_rank'] = gene_scores[['QQratio_rank', 'MS_rank']].min(axis=1).astype(int)
    gene_scores['bimodality_score_rank'][~(
            (gene_scores['log_fold_change'] > log_fold_change_thres) &
            (gene_scores['oncell_size'] > int(oncell_size_min_count)) &
            (gene_scores['oncell_size'] < int(oncell_size_max_proportion / 100 * num_cells))
    )] = len(gene_scores)

    gene_scores['proximity_score_rank'] = gene_scores['proximity_score_rank'].copy().astype(int)
    gene_scores['proximity_score_rank'][~(
            (gene_scores['log_fold_change'] > log_fold_change_thres) &
            (gene_scores['oncell_size'] > int(oncell_size_min_count)) &
            (gene_scores['oncell_size'] < int(oncell_size_max_proportion / 100 * num_cells))
    )] = len(gene_scores)

    MarcoPolo_score = gene_scores[['voting_score_rank', 'proximity_score_rank', 'bimodality_score_rank']].min(axis=1)

    gene_scores['MarcoPolo'] = MarcoPolo_score
    gene_scores['MarcoPolo_rank'] = pd.Series(np.arange(gene_scores.shape[0]),
                                           index=gene_scores.sort_values(['MarcoPolo', 'log_fold_change'],
                                                                      ascending=[True, False]).index).loc[
        gene_scores.index]

    gene_scores = gene_scores.reindex(sorted(gene_scores.columns), axis=1)

    return gene_scores
    
    
