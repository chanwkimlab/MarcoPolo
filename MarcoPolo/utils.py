import multiprocessing
import pickle

import numpy as np
import pandas as pd


def gamma_argmax_list_to_minorsize_list_list(gamma_argmax_list):
    minorsize_list=np.sum(gamma_argmax_list==0,axis=1)
    print(minorsize_list)
    minorsize_list_list=[np.clip(minorsize_list, a_min=0, a_max=minorsize) for minorsize in minorsize_list]

    return np.array(minorsize_list_list)


def gamma_argmax_list_to_intersection(gamma_argmax_list,idx):
    intersection=np.sum((gamma_argmax_list[idx]==gamma_argmax_list)&(gamma_argmax_list[idx]==0),axis=1)
    return intersection


def gamma_argmax_list_to_intersection_list(gamma_argmax_list):
    pool=multiprocessing.Pool(processes=16)

    intersection_list=pool.starmap(gamma_argmax_list_to_intersection,[(gamma_argmax_list,i) for i in np.arange(gamma_argmax_list.shape[0])])

    pool.close()
    pool.join()

    return np.array(intersection_list)


def gamma_exp_to_gamma_argmax(gamma,exp_data_select=None):
    gamma_argmax=np.argmax(gamma,axis=1)
    gamma_argmax_counts=list(np.unique(gamma_argmax,return_counts=True))
    if exp_data_select is None:
        key_newkey=pd.DataFrame(gamma_argmax_counts,index=['idx','counts']).T.set_index('idx').sort_values(by='counts',ascending=True).index.tolist()
    else:
        gamma_argmax_counts_lfc=gamma_argmax_counts+        [(list(map(lambda x: np.mean(exp_data_select[gamma_argmax==x],axis=0)-np.mean(exp_data_select[gamma_argmax!=x],axis=0), gamma_argmax_counts[0])))]
        key_newkey=pd.DataFrame(gamma_argmax_counts_lfc,index=['idx','counts','lfc']).T.astype({'idx':int,'counts':int}).set_index('idx').sort_values(by='lfc',ascending=False).index.tolist()
    gamma_argmax=[key_newkey.index(argmax) for argmax in gamma_argmax]

    return gamma_argmax


def gamma_list_exp_data_to_gamma_argmax_list(gamma_list,exp_data=None):

    pool=multiprocessing.Pool(processes=16)

    if exp_data is None:
        gamma_argmax_list=pool.map(gamma_exp_to_gamma_argmax,gamma_list)
    else:
        gamma_argmax_list=pool.starmap(gamma_exp_to_gamma_argmax,[(gamma_list[i],exp_data[i]) for i in range(len(gamma_list))])

    pool.close()
    pool.join()

    return np.array(gamma_argmax_list)


def read_QQscore(path,num_cluster_list=[1,2]):
    """
    read QQ
    :param path str: input/output path
    """
    result_list=[]
    gamma_list_list=[]
    delta_log_list_list=[]
    beta_list_list=[]

    for num_cluster_list_idx,num_cluster in enumerate(num_cluster_list):
        result_list.append(pd.read_csv('{}.QQscore.{}.tsv'.format(path,num_cluster),sep='\t',index_col=0))
        with open('{}.QQscore.{}.gamma_list.pickle'.format(path,num_cluster), 'rb') as f:
            gamma_list_list.append(pickle.load(f))
        with open('{}.QQscore.{}.delta_log_list.pickle'.format(path,num_cluster), 'rb') as f:
            delta_log_list_list.append(pickle.load(f))
        with open('{}.QQscore.{}.beta_list.pickle'.format(path,num_cluster), 'rb') as f:
            beta_list_list.append(pickle.load(f))

    return result_list,gamma_list_list,delta_log_list_list,beta_list_list