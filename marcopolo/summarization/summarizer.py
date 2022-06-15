import pickle
    
import numpy as np
import pandas as pd    

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.io import mmread
from sklearn.decomposition import PCA

import MarcoPolo.QQscore as QQ

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter


def save_MarcoPolo(input_path,output_path,mode=2,voting_thres=0.7,norm_thres=10,n_pc=2,lfc_thres=0.6,minor_size_min=10,minor_size_max_percent=70):
    
    """
    Save MarcoPolo result
    
    :param input_path str: input file path
    :param output_path str: output file path
    :param mode int: default=2
    :param voting thres float: value >=0 and <=1
    :param n_pc int: value >=1 and <=50
    """
    lfc_thres
    # read scRNA data
    exp_data=mmread('{}.data.counts.mm'.format(input_path)).toarray().astype(float)
    with open('{}.data.col'.format(input_path),'r') as f: exp_data_col=[i.strip().strip('"') for i in f.read().split()]
    with open('{}.data.row'.format(input_path),'r') as f: exp_data_row=[i.strip().strip('"') for i in f.read().split()]
    assert exp_data.shape==(len(exp_data_row),len(exp_data_col))
    assert len(set(exp_data_row))==len(exp_data_row)
    assert len(set(exp_data_col))==len(exp_data_col)        

    exp_data_meta=pd.read_csv('{}.metadatacol.tsv'.format(input_path),sep='\t')

    cell_size_factor=pd.read_csv('{}.size_factor.tsv'.format(input_path),sep='\t',header=None)[0].values.astype(float)#.reshape(-1,1)

    x_data_intercept=np.array([np.ones(exp_data.shape[1])]).transpose()
    x_data_null=np.concatenate([x_data_intercept],axis=1)
    
    #read QQ
    result_list,gamma_list_list,delta_log_list_list,beta_list_list=QQ.read_QQscore(input_path,[1,mode])
    gamma_list=gamma_list_list[-1]
    
    gamma_argmax_list=QQ.gamma_list_exp_data_to_gamma_argmax_list(gamma_list,exp_data)
    
    
    # mean_0_all    
    mean_all=np.array([np.mean(exp_data[i,:]) for i in range(gamma_argmax_list.shape[0])])
    mean_0=np.array([np.mean(exp_data[i,gamma_argmax_list[i]==0]) for i in range(gamma_argmax_list.shape[0])])    
    mean_0_all=mean_0-mean_all    
    
    # QQratio
    QQratio=result_list[0]['Q']/result_list[-1]['Q']
    
    # voting score
    minorsize_list=np.sum(gamma_argmax_list==0,axis=1)
    minorsize_cliplist=QQ.gamma_argmax_list_to_minorsize_list_list(gamma_argmax_list)
    intersection_list=QQ.gamma_argmax_list_to_intersection_list(gamma_argmax_list)
    intersectioncount_prop=((intersection_list/minorsize_cliplist))
    intersectioncount_threshold=((intersection_list/minorsize_cliplist)>voting_thres)
    intersectioncount_thresholdcount=np.sum(intersectioncount_threshold,axis=1)
    
    ##########
    
    # lfc
    lfc=np.log10(np.array([np.mean(exp_data[i,gamma_argmax_list[i]==0],axis=0) for i in range(gamma_argmax_list.shape[0])])/                np.array([np.mean(exp_data[i,gamma_argmax_list[i]!=0],axis=0) for i in range(gamma_argmax_list.shape[0])]))      
    
    # PC Variance    
    exp_data_norm=np.log1p(10000*exp_data/exp_data.sum(axis=0))
    exp_data_norm_scale=(exp_data_norm-exp_data_norm.mean(axis=1).reshape(-1,1))/    exp_data_norm.std(axis=1).reshape(-1,1)
    exp_data_norm_scale[exp_data_norm_scale>norm_thres]=norm_thres

    pca = PCA(n_components=50)
    pca.fit(exp_data_norm_scale.T)
    exp_data_norm_scale_pc=pca.transform(exp_data_norm_scale.T)
    exp_data_norm_scale_pc.shape
               
    exp_data_norm_scale_pc_topstdmean=np.array([exp_data_norm_scale_pc[gamma_argmax_list[i]==0,:n_pc].std(axis=0).mean() for i in range(gamma_argmax_list.shape[0])])
        
        
    try:
        markerrho=pd.read_csv('{}.markerrho.tsv'.format(input_path),index_col=0,sep='\t')
    except:
        print('markerrho does not exist')
        markerrho=pd.DataFrame([])  
        #except NameError:
    try:
        maxdiff=pd.read_csv('{}.maxdiff.tsv'.format(input_path),index_col=0,header=None,sep='\t')[1]
    except:
        print('maxdiff does not exist')
        maxdiff=np.zeros_like(QQratio.values)
        assert maxdiff.shape[0]==len(exp_data_row)          
             

    allscore=pd.DataFrame([QQratio.values,
                           intersectioncount_thresholdcount,
                           exp_data_norm_scale_pc_topstdmean,
                           lfc,
                           mean_0_all,
                           minorsize_list,
                           maxdiff,
                           list(map(lambda x: x in markerrho.columns,exp_data_row))
                          ],
                          index=['QQratio',
                                 'intersectioncount',
                                 'PCstd',
                                 'lfc',
                                 'mean_0_all',
                                 'minorsize',
                                 'maxdiff',
                                 'ismarker']).T

    allscore['QQratio_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['QQratio'].sort_values(ascending=False).index).loc[allscore.index]

    allscore['intersectioncount_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['intersectioncount'].sort_values(ascending=False).index).loc[allscore.index]
    
    allscore['PCstd_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['PCstd'].sort_values(ascending=True).index).loc[allscore.index]
    
    allscore['lfc_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['lfc'].sort_values(ascending=False).index).loc[allscore.index]
    
    allscore['mean_0_all_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['mean_0_all'].sort_values(ascending=False).index).loc[allscore.index]
    allscore['minorsize_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore['minorsize'].sort_values(ascending=False).index).loc[allscore.index]

    allscore['intersectioncount_rank'][allscore['intersectioncount']==0]=499999
    allscore['intersectioncount_rank'][allscore['intersectioncount']==1]=999999

    
    allscore['votingscore_rank']=allscore['intersectioncount_rank'].copy()
    allscore['votingscore_rank'][~(
            (allscore['lfc']>lfc_thres)&
            (allscore['minorsize']>int(minor_size_min))&
            (allscore['minorsize']<int(minor_size_max_percent/100*len(exp_data_col)))
            )]=len(allscore)    
    
    allscore['bimodalityscore_rank']=allscore[['QQratio_rank','mean_0_all_rank']].min(axis=1)
    allscore['bimodalityscore_rank'][~(
            (allscore['lfc']>lfc_thres)&
            (allscore['minorsize']>int(minor_size_min))&
            (allscore['minorsize']<int(minor_size_max_percent/100*len(exp_data_col)))
            )]=len(allscore)    
    
    allscore['proximityscore_rank']=allscore['PCstd_rank'].copy()
    allscore['proximityscore_rank'][~(
            (allscore['lfc']>lfc_thres)&
            (allscore['minorsize']>int(minor_size_min))&
            (allscore['minorsize']<int(minor_size_max_percent/100*len(exp_data_col)))
            )]=len(allscore)       
    
    
    MarcoPolo_score=allscore[['votingscore_rank','proximityscore_rank','bimodalityscore_rank']].min(axis=1)
    
    
    allscore['MarcoPolo']=MarcoPolo_score
    allscore['MarcoPolo_rank']=pd.Series(np.arange(allscore.shape[0]),index=allscore.sort_values(['MarcoPolo','lfc'],ascending=[True,False]).index).loc[allscore.index]

    
    allscore.to_csv('{path}.MarcoPolo.{mode}.rank.tsv'.format(path=output_path,mode=mode),sep='\t')
    #np.save('{path}.MarcoPolo.{mode}.rank.npy'.format(path=output_path,mode=mode), intersectioncount_prop)
    
    
    base = importr('base')

    with localconverter(ro.default_converter + pandas2ri.converter):
        allscore_r = ro.conversion.py2rpy(allscore.fillna(0))
        ro.r.assign("result", allscore_r)

    with localconverter(ro.default_converter + pandas2ri.converter):
        gamma_argmax_list_r = ro.conversion.py2rpy(pd.DataFrame(gamma_argmax_list))
        ro.r.assign("gamma_argmax_list", gamma_argmax_list_r)        
        
    base.save_image('{path}.MarcoPolo.{mode}.RData'.format(path=output_path,mode=mode))