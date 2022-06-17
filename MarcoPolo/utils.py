import multiprocessing
import warnings

import numpy as np
import pandas as pd


def gamma_argmax_list_to_oncell_size_list_list(gamma_argmax_list: np.ndarray)->np.ndarray:
    """

    Args:
        gamma_argmax_list: List of gamma_argmax.

    Returns:
        np.ndarray: List of oncell size.

    """
    oncellsize_list=np.sum(gamma_argmax_list==0,axis=1)
    # print(oncellsize_list)
    oncellsize_list_list=[np.clip(oncellsize_list, a_min=0, a_max=oncellsize) for oncellsize in oncellsize_list]

    return np.array(oncellsize_list_list)


def gamma_argmax_list_to_intersection(gamma_argmax_list, idx):
    """

    Args:
        gamma_argmax_list: List of gamma_argmax.
        idx: Index of gamma_argmax_list.

    Returns:
        np.ndarray: Intersection.

    """
    intersection = np.sum((gamma_argmax_list[idx] == gamma_argmax_list) & (gamma_argmax_list[idx] == 0), axis=1)
    return intersection

def gamma_argmax_list_to_intersection_list(gamma_argmax_list: np.ndarray)->np.ndarray:
    """

    Args:
        gamma_argmax_list:  List of gamma_argmax.

    Returns:
        np.ndarray: List of intersection.

    """

    pool=multiprocessing.Pool(processes=16)

    intersection_list=pool.starmap(gamma_argmax_list_to_intersection,[(gamma_argmax_list,i) for i in np.arange(gamma_argmax_list.shape[0])])

    pool.close()
    pool.join()

    return np.array(intersection_list)

def gamma_expression_to_gamma_argmax(gamma: np.ndarray, expression: np.ndarray = None) -> np.ndarray:
    """

    Args:
        gamma: A gamma matrix.
        expression: If expression is not None, it is used to calculate the which group has higher expression mean.

    Returns:
        np.ndarray: gamma_argmax.

    """
    gamma_argmax = np.argmax(gamma, axis=1)
    gamma_argmax_counts = list(np.unique(gamma_argmax, return_counts=True))
    if expression is None:
        key_newkey = pd.DataFrame(gamma_argmax_counts, index=['idx', 'counts']).T.set_index('idx').sort_values(
            by='counts', ascending=True).index.tolist()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            gamma_argmax_counts_lfc = gamma_argmax_counts + [(list(
                map(lambda x: np.mean(expression[gamma_argmax == x], axis=0) - np.mean(
                    expression[gamma_argmax != x], axis=0), gamma_argmax_counts[0])))]
        key_newkey = pd.DataFrame(gamma_argmax_counts_lfc, index=['idx', 'counts', 'lfc']).T.astype(
            {'idx': int, 'counts': int}).set_index('idx').sort_values(by='lfc', ascending=False).index.tolist()
    gamma_argmax = [key_newkey.index(argmax) for argmax in gamma_argmax]

    return gamma_argmax

def gamma_list_expression_matrix_to_gamma_argmax_list(gamma_list: np.ndarray, expression_matrix: np.ndarray=None)->np.ndarray:
    """

    Args:
        gamma_list: List of gamma matrices.
        expression_matrix: If expression_matrix is not None, it is used to calculate the which group has higher expression mean.

    Returns:
        List of gamma_argmax.

    """
    pool=multiprocessing.Pool(processes=16)

    if expression_matrix is None:
        gamma_argmax_list=pool.map(gamma_expression_to_gamma_argmax, gamma_list)
    else:
        gamma_argmax_list=pool.starmap(gamma_expression_to_gamma_argmax, [(gamma_list[i], expression_matrix[:, i]) for i in range(len(gamma_list))])

    pool.close()
    pool.join()

    return np.array(gamma_argmax_list)