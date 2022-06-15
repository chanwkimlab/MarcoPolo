import pickle
import sys
import os
    
import numpy as np
import pandas as pd    

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.io import mmread
from sklearn.decomposition import PCA

import MarcoPolo.QQscore as QQ

import matplotlib.pyplot as plt

import shutil
from jinja2 import Template
import rpy2


#Porting of https://github.com/satijalab/seurat/blob/b51801bc4b1a66aed5456473c9fe0be884994c93/R/visualization.R#L2686
def DiscretePalette(n, palette=None):
    """
    Generate a list of colors that are well separated with one another.
    
    :param int n: number of colors
    
    :return numpy.array: list of colors
    """
    palettes={
                'alphabet':[
                  "#F0A0FF", "#0075DC", "#993F00", "#4C005C", "#191919", "#005C31",
                  "#2BCE48", "#FFCC99", "#808080", "#94FFB5", "#8F7C00", "#9DCC00",
                  "#C20088", "#003380", "#FFA405", "#FFA8BB", "#426600", "#FF0010",
                  "#5EF1F2", "#00998F", "#E0FF66", "#740AFF", "#990000", "#FFFF80",
                  "#FFE100", "#FF5005"
                ],
                'alphabet2':[
                  "#AA0DFE", "#3283FE", "#85660D", "#782AB6", "#565656", "#1C8356",
                  "#16FF32", "#F7E1A0", "#E2E2E2", "#1CBE4F", "#C4451C", "#DEA0FD",
                  "#FE00FA", "#325A9B", "#FEAF16", "#F8A19F", "#90AD1C", "#F6222E",
                  "#1CFFCE", "#2ED9FF", "#B10DA1", "#C075A6", "#FC1CBF", "#B00068",
                  "#FBE426", "#FA0087"
                ],
                'glasbey':[
                  "#0000FF", "#FF0000", "#00FF00", "#000033", "#FF00B6", "#005300",
                  "#FFD300", "#009FFF", "#9A4D42", "#00FFBE", "#783FC1", "#1F9698",
                  "#FFACFD", "#B1CC71", "#F1085C", "#FE8F42", "#DD00FF", "#201A01",
                  "#720055", "#766C95", "#02AD24", "#C8FF00", "#886C00", "#FFB79F",
                  "#858567", "#A10300", "#14F9FF", "#00479E", "#DC5E93", "#93D4FF",
                  "#004CFF", "#F2F318"
                ],
                'polychrome':[
                  "#5A5156", "#E4E1E3", "#F6222E", "#FE00FA", "#16FF32", "#3283FE",
                  "#FEAF16", "#B00068", "#1CFFCE", "#90AD1C", "#2ED9FF", "#DEA0FD",
                  "#AA0DFE", "#F8A19F", "#325A9B", "#C4451C", "#1C8356", "#85660D",
                  "#B10DA1", "#FBE426", "#1CBE4F", "#FA0087", "#FC1CBF", "#F7E1A0",
                  "#C075A6", "#782AB6", "#AAF400", "#BDCDFF", "#822E1C", "#B5EFB5",
                  "#7ED7D1", "#1C7F93", "#D85FF7", "#683B79", "#66B0FF", "#3B00FB"
                ],
                'stepped':[
                  "#990F26", "#B33E52", "#CC7A88", "#E6B8BF", "#99600F", "#B3823E",
                  "#CCAA7A", "#E6D2B8", "#54990F", "#78B33E", "#A3CC7A", "#CFE6B8",
                  "#0F8299", "#3E9FB3", "#7ABECC", "#B8DEE6", "#3D0F99", "#653EB3",
                  "#967ACC", "#C7B8E6", "#333333", "#666666", "#999999", "#CCCCCC"
                ]
            }
    if palette is None:
        if n<=26:
            palette="alphabet"
        elif n<=32:
            palette="glasbey"
        else:
            palette="polychrome"
    
    palette_array= palettes[palette]
    assert n<=len(palette_array), "Not enough colours in specified palette"

    return np.array(palette_array)[np.arange(n)]



def generate_report(input_path,output_path,top_num_table=1000,top_num_figure=1000,output_mode='pub',gene_info_path='https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz',mode=2):
    """
    Generate HTML report.
    
    :param input_path str: input file path
    :param output_path str: output file path
    :param top_num_table int: number of genes to be shown in HTML report
    :param top_num_figure int: number of genes to make plots
    :param output_mode str: 'pub' is preferred
    :param gene_info_path str: gene info path
    :param mode int: default=2
    
    """        
    
    path=input_path
    report_path=output_path
    
    
    print("------Loading dataset------")
    exp_data=mmread('{}.data.counts.mm'.format(path)).toarray().astype(float)
    with open('{}.data.col'.format(path),'r') as f: exp_data_col=[i.strip().strip('"') for i in f.read().split()]
    with open('{}.data.row'.format(path),'r') as f: exp_data_row=[i.strip().strip('"') for i in f.read().split()]
    assert exp_data.shape==(len(exp_data_row),len(exp_data_col))
    assert len(set(exp_data_row))==len(exp_data_row)
    assert len(set(exp_data_col))==len(exp_data_col)        

    exp_data_meta=pd.read_csv('{}.metadatacol.tsv'.format(path),sep='\t')

    cell_size_factor=pd.read_csv('{}.size_factor.tsv'.format(path),sep='\t',header=None)[0].values.astype(float)#.reshape(-1,1)

    x_data_intercept=np.array([np.ones(exp_data.shape[1])]).transpose()
    x_data_null=np.concatenate([x_data_intercept],axis=1)
    
    
    result_list,gamma_list_list,delta_log_list_list,beta_list_list=QQ.read_QQscore(path,[1,mode])
    
    gamma_list=gamma_list_list[-1]    

    gamma_argmax_list=QQ.gamma_list_exp_data_to_gamma_argmax_list(gamma_list,exp_data)#gamma_argmax_list=QQ.gamma_list_to_gamma_argmax_list(gamma_list)
    gamma_argmax_list,gamma_argmax_list.shape
    
    
    # voting score
    minorsize_list=np.sum(gamma_argmax_list==0,axis=1)
    minorsize_cliplist=QQ.gamma_argmax_list_to_minorsize_list_list(gamma_argmax_list)
    intersection_list=QQ.gamma_argmax_list_to_intersection_list(gamma_argmax_list)
    intersectioncount_prop=((intersection_list/minorsize_cliplist))
    intersectioncount_prop_top10=[np.arange(0,len(i))[i>=sorted(i)[-10]][:10] for i in intersectioncount_prop]
    intersectioncount_threshold=((intersection_list/minorsize_cliplist)>0.7)
    intersectioncount_thresholdcount=np.sum(intersectioncount_threshold,axis=1)
    
    allscore=pd.read_csv('{}.MarcoPolo.{}.rank.tsv'.format(path,mode),index_col=0,sep='\t')
    
    allscore_munge=allscore.copy()
    allscore_munge['Gene ID']=exp_data_row
    
    allscore_munge['Voting_genes_top10']=[allscore_munge['Gene ID'][i].values for i in intersectioncount_prop_top10]
    allscore_munge_voting=allscore_munge.copy()    
    
    if gene_info_path is not None:
        print("------Annotating genes------")
        
        gene_info=pd.read_csv(gene_info_path,sep='\t')

        by='ID' if 'ENS' in exp_data_row[0] else 'name'

        gene_info_select_list=[]

        column_list=['Symbol','description','Other_designations','type_of_gene','dbXrefs']

        for idx, query in enumerate(exp_data_row):
            if by=='ID':
                gene_info_select=gene_info[gene_info['dbXrefs'].str.contains(query,regex=False)]
            else:
                gene_info_select=gene_info[gene_info['Symbol'].str.lower()==query.lower()]
                if len(gene_info_select)==0:
                    gene_info_select=gene_info[gene_info['Synonyms'].str.lower().str.contains(query.lower(),regex=False)]

            if len(gene_info_select)>=1:
                gene_info_select_list.append(gene_info_select[column_list].iloc[0])
            else:
                gene_info_select_list.append(pd.Series(index=column_list))
                print(query,len(gene_info_select))

            if idx%100==0:
                sys.stdout.write('\r%0.2f%%' % (100.0 * (idx/len(exp_data_row))))
                sys.stdout.flush()
        gene_info_extract=pd.DataFrame(gene_info_select_list,index=np.arange(len(exp_data_row))) 
    
        assert len(gene_info_extract)==len(allscore_munge)
        allscore_munge=allscore_munge.merge(right=gene_info_extract,left_index=True,right_index=True)   
    #allscore_munge.to_csv('{}.MarcoPolo.{}.rank.munge.tsv'.format(path,mode),sep='\t')
    
    allscore_munge['img']=allscore_munge.apply(lambda x: '<img src="plot_image/{idx}.png" alt="{idx}">'.format(idx=x.name),axis=1)
    
    allscore_munge['Log2FC']=allscore_munge['lfc']/np.log10(2)
    if output_mode=='report':
        
        allscore_munge=allscore_munge[[
                                    'MarcoPolo_rank',
                                    'Gene ID','Symbol',
                                    'description', 'Other_designations', 'type_of_gene',
                                    'Log2FC',
                                    'MarcoPolo',
                                    'QQratio', 'QQratio_rank',
                                    'QQdiff', 'QQdiff_rank',
                                    'votingscore', 'votingscore_rank',
                                    'mean_0_all','mean_0_all_rank',
                                    'PCvariance', 'PCvariance_rank',
                                    'lfc', 'lfc_rank',
                                    'minorsize','minorsize_rank',
                                    'dbXrefs','img'
                                   ]]

        allscore_munge[['Log2FC',
                        'QQratio', 
                        'QQdiff', 
                        'votingscore', 'votingscore_rank',
                        'mean_0_all',
                        'PCvariance',
                        'lfc']]=\
        allscore_munge[['Log2FC',
                        'QQratio',
                        'QQdiff', 
                        'votingscore', 'votingscore_rank',
                        'mean_0_all',
                        'PCvariance',
                        'lfc']].round(2)

    elif output_mode=='pub':
        if gene_info_path is None:
            allscore_munge=allscore_munge[[
                                        'MarcoPolo_rank',
                                        'Gene ID',
                                        'Log2FC',
                                        'MarcoPolo',
                                        'bimodalityscore_rank',
                                        'votingscore_rank',
                                        'proximityscore_rank',
                                        'lfc', 'lfc_rank',
                                        'minorsize','minorsize_rank',
                                        'img'
                                       ]]  
        else:
            allscore_munge=allscore_munge[[
                                        'MarcoPolo_rank',
                                        'Gene ID','Symbol',
                                        'description', 'Other_designations', 'type_of_gene',
                                        'Log2FC',
                                        'MarcoPolo',
                                        'bimodalityscore_rank',
                                        'votingscore_rank',
                                        'proximityscore_rank',
                                        'lfc', 'lfc_rank',
                                        'minorsize','minorsize_rank',
                                        'dbXrefs','img'
                                       ]]            
            
          

        allscore_munge[['Log2FC',
                        'lfc']]=\
        allscore_munge[['Log2FC',
                        'lfc']].round(2)    

    else:
        raise



    #shutil.copy('report/template/index.html', 'report/{}/index.html'.format(dataset_name_path))
    
    os.makedirs('{}'.format(report_path),exist_ok=True)
    os.makedirs('{}/plot_image'.format(report_path),exist_ok=True)
    os.makedirs('{}/assets'.format(report_path),exist_ok=True)
    
    shutil.copy(os.path.join(os.path.dirname(__file__), 'template/assets/scripts.js'), '{}/assets/scripts.js'.format(report_path))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'template/assets/styles.css'), '{}/assets/styles.css'.format(report_path))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'template/assets/details_open.png'), '{}/assets/details_open.png'.format(report_path))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'template/assets/details_close.png'), '{}/assets/details_close.png'.format(report_path))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'template/assets/mp.png'), '{}/assets/mp.png'.format(report_path))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'template/assets/mp_white.png'), '{}/assets/mp_white.png'.format(report_path))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'template/assets/mp_white_large_font.png'), '{}/assets/mp_white_large_font.png'.format(report_path))
    
    
    

    with open(os.path.join(os.path.dirname(__file__), 'template/index.html'), 'r') as f:
        template_read=f.read()
    template = Template(source=template_read)  

    MarcoPolo_table=allscore_munge.sort_values("MarcoPolo_rank",ascending=True).set_index('MarcoPolo_rank').iloc[:top_num_table]
    MarcoPolo_table.index+=1
    MarcoPolo_table=MarcoPolo_table.to_html(classes="table table-bordered",table_id='dataTable')

    MarcoPolo_table=MarcoPolo_table.replace('<table ','<table width="100%" cellspacing="0" ')
    template_rendered=template.render(MarcoPolo_table=MarcoPolo_table,num_gene=exp_data.shape[0],num_cell=exp_data.shape[1])

    with open('{}/index.html'.format(report_path),'w') as f:
        f.write(template_rendered)    
        
    allscore_munge_voting[['Gene ID','Voting_genes_top10']].to_html('{}/voting.html'.format(report_path))    
    print("------Drawing figures------")
    
    exp_data_meta_transformed=exp_data_meta.copy()
    
    
    
    import seaborn as sns

    plt.rcParams["figure.figsize"] = (16,16)
    plt.rcParams["font.size"] = 10
    plt.rcParams['font.family']='Arial'

    fig = plt.figure(figsize=(10, 10))
    gs=fig.add_gridspec(10,10)

    ax=fig.add_subplot(gs[1:9,1:9])

    #ax = fig.add_subplot(111)

    plot_value=exp_data_meta_transformed['phenoid']
    plot_value_unique=plot_value.unique().tolist()
    plot_value_int=list(map(lambda x: plot_value_unique.index(x),plot_value))


    #sns.scatterplot(x="tSNE_1", y="tSNE_2",hue=plot_value,style=np.array((list(range(0,2))*30))[plot_value_int],data=exp_data_meta,palette=plt.cm.rainbow if plot_value.dtype==int else None)#,)#,s=40,palette=plt.cm.rainbow)#,linewidth=0.3)    
    scatterfig=sns.scatterplot(x="TSNE_1", y="TSNE_2",hue=plot_value,data=exp_data_meta_transformed,
                    palette=DiscretePalette(len(plot_value_unique)).tolist() if plot_value.dtype==int else None,
                   ax=ax,s=25,alpha=1,edgecolor='None'
                   )#,)#,s=40,palette=plt.cm.rainbow)#,linewidth=0.3)    


    ax.get_legend().remove()
    ax.set_ylabel(' ')
    ax.set_xlabel(' ')

    ax.set_xticks([])
    ax.set_yticks([])
    #ax.get_xaxis().set_ticks([])
    #ax.get_xaxis().set_label('a')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    #ax.legend('')
    #
    plt.savefig('{}/plot_image/2D_Plot.png'.format(report_path),dpi=100,bbox_inches='tight')
    plt.show()
    


    #%matplotlib inline
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    #plt.rcParams["figure.figsize"] = (5*8,10)
    plt.rcParams["font.size"] = 15
    plt.rcParams['font.family']='Arial'
    plt.ioff()

    #for idx,(iter_idx,value) in enumerate(marker_criteria.iteritems()):
    #for idx, (iter_idx,row) in enumerate(allscore[allscore['ismarker']==True].sort_values('all_rank').iterrows()):    
    exp_data_corrected=(exp_data/cell_size_factor)
    for count_idx, (iter_idx,row) in enumerate(allscore.sort_values('MarcoPolo',ascending=True).iloc[:].iterrows()):    



        #subplot_size=(1,1+1+1+1+1)
        subplot_size=(1,1+1+1+1)


        #if idx==10*(subplot_size[0]*subplot_size[1]):
        #    break    


        #fig, axes = plt.subplots(*subplot_size)
        #fig = plt.figure(figsize=(3*10, 3*2)) 
        fig = plt.figure(figsize=(3*8, 6)) 
        gs=fig.add_gridspec(6,3*8)
        subplot_list=[fig.add_subplot(gs[0:6,0:6]),
                      fig.add_subplot(gs[0:6,6+2:6+2+6])]

        #gs = gridspec.GridSpec(subplot_size[0], subplot_size[1], width_ratios=[2, 1, 1, 1]) 


        #plt.subplots_adjust(wspace=0, hspace=0)

        #plt.text(.5,.95,'QQ:{:.1f}={}th Voting:{:d}={:d}th->{:d}'.format(row['QQratio'],int(row['QQratio_rank']),int(row['votingscore']),int(row['votingscore_rank']),int(row['all_rank']),idx),
        #                horizontalalignment='center',
        #                transform=ax.transAxes)


        exp_data_corrected_on=exp_data_corrected[iter_idx][gamma_argmax_list[iter_idx]==0]
        exp_data_corrected_off=exp_data_corrected[iter_idx][gamma_argmax_list[iter_idx]==1]

        bins_log=[np.power(1.2,i) for i in range(np.max([1,int(np.log(np.max([1,np.max(exp_data_corrected)]))/np.log(1.2))]))]
        bins_log_on=[np.power(1.1,i) for i in range(
            np.max([1,int(np.log(np.max([1,np.min(exp_data_corrected_on)]))/np.log(1.1))]),
            np.max([1,int(np.log(np.max([1,np.max(exp_data_corrected_on)]))/np.log(1.1))])

        )]
        bins_log_off=[np.power(1.2,i) for i in range(np.max([1,int(np.log(np.max([1,np.max(exp_data_corrected_off)]))/np.log(1.2))]))]





        for idx in range(2):
            #ax=plt.subplot(gs[idx%((subplot_size[0]*subplot_size[1]))])
            ax=subplot_list[idx]
            #ax=axes.flatten()[idx%((subplot_size[0]*subplot_size[1]))]
            #ax.set_axis_off()
            #ax.set_xticklabels([])
            #ax.set_yticklabels([])
            #ax.legend().remove()
            #ax.legend().set_visible(False)
            if idx==-1:
                plot_value=np.log10(1+exp_data_corrected[iter_idx])
                #plot_value=(plot_value-np.min(plot_value))/np.var(plot_value)
                #gamma_argmax_list[iter_idx]

                #ax.set_title(label=)
                #ax.title.set_text('QQ:{:.1f}={}th Voting:{:d}={:d}th->{:d}={:d}th'.format(row['QQscore'],int(row['QQscore_rank']),int(row['votingscore']),int(row['votingscore_rank']),int(row['all_rank']),idx))
                ax.title.set_text('tSNE')
                sns.scatterplot(x="TSNE_1", y="TSNE_2",label=None,legend=None,
                                hue=plot_value,data=exp_data_meta_transformed,ax=ax,s=15,linewidth=0.3,alpha=0.4,palette=plt.cm.Blues)#,palette=plt.cm.rainbow)#,linewidth=0.3)            
            if idx==0:

                #ax.set_title(label=)
                #ax.title.set_text('QQ:{:.1f}={}th Voting:{:d}={:d}th->{:d}={:d}th'.format(row['QQscore'],int(row['QQscore_rank']),int(row['votingscore']),int(row['votingscore_rank']),int(row['all_rank']),idx))

                plot_value=exp_data_meta_transformed['phenoid']
                plot_value_unique=plot_value.unique().tolist()
                plot_value_int=list(map(lambda x: plot_value_unique.index(x),plot_value))            

                s=10
                sns.scatterplot(x="TSNE_1", y="TSNE_2",hue=plot_value,data=exp_data_meta_transformed,
                                palette=DiscretePalette(len(plot_value_unique)).tolist() if plot_value.dtype==int else None,
                               ax=ax,alpha=0.3,edgecolor="None",
                                s=s
                               )
                sns.scatterplot(x="TSNE_1", y="TSNE_2",data=exp_data_meta_transformed.loc[gamma_argmax_list[iter_idx]==0],ax=ax,
                                edgecolor=[1,0,0,1],
                                facecolors="None",
                                linewidth=1,
                                s=s,
                                zorder=10
                               )

                """
                plot_value=gamma_argmax_list[iter_idx]
                sns.scatterplot(x="tSNE_1", y="tSNE_2", label=None,legend=None,hue=plot_value,
                                data=data,ax=ax,s=15,linewidth=0.3,alpha=0.4)#,palette=plt.cm.rainbow)#,linewidth=0.3)            
                """            

                ax.title.set_text('On-Off in 2D plot')
                ax.get_legend().remove()

                ax.set_xlabel('Dim 1')            
                ax.set_ylabel('Dim 2',labelpad=-10)

                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(1.5)              

            elif idx==1:
                ax.title.set_text('Expression of Cells')
                #bins_count,bins,patch=ax.hist(exp_data[iter_idx],bins=bins_log,color='black')
                ax.hist(exp_data_corrected_on,bins=bins_log,label='On',color=(1,0,0,0.8))
                ax.hist(exp_data_corrected_off,bins=bins_log,label='Off',color=(0.5,0.5,0.5,0.5))
                ax.set_xscale('log')

                ax.set_ylabel('Cell Counts')                        
                ax.set_xlabel('Expression count (size factor corrected)')

                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False) 

                leg=ax.legend(loc='upper left',
                                    fontsize=15,
                                    frameon=False,
                                    bbox_to_anchor=(0.22, -0.15),
                                    ncol=2,
                                  handletextpad=0.2,
                                      columnspacing=1.3,
                                    markerscale=2.5)  
                [rec.set_height(8) for rec in leg.get_patches()]
                [rec.set_width(15) for rec in leg.get_patches()]
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(2)             

                #sns.distplot(a=exp_data[iter_idx],kde=False,color='black',ax=ax)
            elif idx==2:
                ax.title.set_text('Expression of On Cells')
                ax.hist(exp_data_corrected_on,bins=bins_log_on,color=(1,0,0,0.8))
                #bins_count,bins,patch=ax.hist(exp_data_on,bins=bins_log,color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
                ax.set_xscale('log')

                ax.set_ylabel('Cell Counts')                        
                ax.set_xlabel('Expression count (size factor corrected)')

                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)              

                #sns.distplot(a=exp_data[iter_idx][gamma_argmax_list[iter_idx]==0],kde=False,color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],ax=ax)
            elif idx==3:
                ax.title.set_text('Expression of Off Cells')
                ax.hist(exp_data_corrected_off,bins=bins_log_off,color=(0.5,0.5,0.5,0.5))
                #bins_count,bins,patch=ax.hist(exp_data_off,bins=bins_log,color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
                ax.set_xscale('log')

                ax.set_ylabel('Cell Frequency')                        
                ax.set_xlabel('Expression count (size factor corrected)')

                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)              

                #sns.distplot(a=exp_data[iter_idx][gamma_argmax_list[iter_idx]!=0],kde=False,color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],ax=ax)
                #ax.axvline(x=1,linewidth=10)
                # plt.Line2D([0.5,0.5],[1,1], transform=fig.transFigure, color="black")
                # ax.axvspan(0.8, 0.9, transform=fig.transFigure,clip_on=False)

        plt.savefig('{}/plot_image/{}.png'.format(report_path,iter_idx),dpi=60,bbox_inches='tight')
        #plt.show()
        plt.close(fig)

        if count_idx==top_num_figure+1:
            break



