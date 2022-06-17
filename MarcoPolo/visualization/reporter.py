import os
import shutil

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
from tqdm import tqdm
from typing import Union, List, Tuple
from pathlib import Path
import MarcoPolo.utils

def get_discrete_palette(n, palette=None):
    """
    Porting of https://github.com/satijalab/seurat/blob/b51801bc4b1a66aed5456473c9fe0be884994c93/R/visualization.R#L2686
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


def annotate_gene_info(gene_scores, gene_query_list, gene_info, by):
    """
    Annotate gene_scores matrix with gene info.
    """

    gene_scores = gene_scores.copy()
    gene_info_select_list = []

    column_list = ['Symbol', 'description', 'Other_designations', 'type_of_gene', 'dbXrefs']

    not_matched_list = []
    pbar=tqdm(gene_query_list)
    for idx, query in enumerate(pbar):
        if by == 'ID':
            gene_info_select = gene_info[gene_info['dbXrefs'].str.contains(query, regex=False)]
        else:
            gene_info_select = gene_info[gene_info['Symbol'].str.lower() == query.lower()]
            if len(gene_info_select) == 0:
                gene_info_select = gene_info[gene_info['Synonyms'].str.lower().str.contains(query.lower(), regex=False)]

        if len(gene_info_select) >= 1:
            gene_info_select_list.append(gene_info_select[column_list].iloc[0])
        else:
            gene_info_select_list.append(pd.Series(index=column_list, dtype=float))
            not_matched_list.append(query)
        #pbar.set_description(f"Number of genes unmatched: {len(not_matched_list)}/ {len(gene_query_list)}")
        pbar.set_postfix({'Num. of unmatched genes': len(not_matched_list)})
    print(f"{len(not_matched_list)} not matched genes: {', '.join(not_matched_list[:20])+ ', ...' if len(not_matched_list) > 20 else ', '.join(not_matched_list)}")
    gene_info_extract = pd.DataFrame(gene_info_select_list, index=np.arange(len(gene_query_list)))


    assert len(gene_info_extract) == len(gene_scores), "gene_info_extract and gene_scores have different length"
    gene_scores = gene_scores.merge(right=gene_info_extract, left_index=True, right_index=True)

    return gene_scores


def generate_html_file(output_dir, gene_scores, num_genes, num_cells, top_num_html=1000):
    os.makedirs('{}'.format(output_dir), exist_ok=True)
    os.makedirs('{}/plot_image'.format(output_dir), exist_ok=True)
    os.makedirs('{}/assets'.format(output_dir), exist_ok=True)

    shutil.copy(os.path.join(os.path.dirname(__file__), 'template/assets/scripts.js'),
                '{}/assets/scripts.js'.format(output_dir))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'template/assets/styles.css'),
                '{}/assets/styles.css'.format(output_dir))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'template/assets/details_open.png'),
                '{}/assets/details_open.png'.format(output_dir))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'template/assets/details_close.png'),
                '{}/assets/details_close.png'.format(output_dir))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'template/assets/mp.png'),
                '{}/assets/mp.png'.format(output_dir))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'template/assets/mp_white.png'),
                '{}/assets/mp_white.png'.format(output_dir))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'template/assets/mp_white_large_font.png'),
                '{}/assets/mp_white_large_font.png'.format(output_dir))

    with open(os.path.join(os.path.dirname(__file__), 'template/index.html'), 'r') as f:
        template_read = f.read()
    template = Template(source=template_read)

    MarcoPolo_table = gene_scores.sort_values("MarcoPolo_rank", ascending=True).set_index('MarcoPolo_rank').iloc[
                      :top_num_html]
    MarcoPolo_table.index += 1
    MarcoPolo_table = MarcoPolo_table.to_html(classes="table table-bordered", table_id='dataTable')

    MarcoPolo_table = MarcoPolo_table.replace('<table ', '<table width="100%" cellspacing="0" ')
    #import ipdb; ipdb.set_trace()
    template_rendered = template.render(MarcoPolo_table=MarcoPolo_table, num_gene=num_genes, num_cell=num_cells)

    with open('{}/index.html'.format(output_dir), 'w') as f:
        f.write(template_rendered)

def generate_image_files(adata, size_factor_key, gamma_argmax_list, gene_scores, low_dim_key, output_dir, top_num_image,
                         cell_color_key=None,
                         main_plot_s=25, main_plot_dpi=100, main_plot_font_size=10, main_plot_font_family='Arial',
                         gene_plot_s=10, gene_plot_dpi=60, gene_plot_font_size=15, gene_plot_font_family='Arial',
                         ):
    print("Drawing figures")

    plt.rcParams["figure.figsize"] = (16, 16)
    plt.rcParams["font.size"] = main_plot_font_size
    plt.rcParams['font.family'] = main_plot_font_family

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(10, 10)

    ax = fig.add_subplot(gs[1:9, 1:9])

    if cell_color_key is None:
        plot_value = pd.Series("temp", index=adata.obs.index)
    else:
        plot_value = adata.obs[cell_color_key]
    plot_value_unique = plot_value.unique().tolist()

    expression_matrix= adata.X.copy()

    if not type(expression_matrix) == np.ndarray:
        expression_matrix = expression_matrix.toarray().astype(float)
    else:
        expression_matrix = expression_matrix.astype(float)

    if size_factor_key is not None:
        expression_matrix = (expression_matrix / adata.obs[size_factor_key].values.astype(float)[:, np.newaxis])
        print("size factor corrected")
    else:
        print("No size factor key")

    cell_meta_info=adata.obs.copy()
    cell_meta_info["coord_x"]=adata.obsm[low_dim_key][:,0]
    cell_meta_info["coord_y"] = adata.obsm[low_dim_key][:, 1]
    scatterfig = sns.scatterplot(x="coord_x", y="coord_y", hue=plot_value, data=cell_meta_info,
                                 palette=get_discrete_palette(
                                     len(plot_value_unique)).tolist() if plot_value.dtype == int else None,
                                 ax=ax, s=main_plot_s, alpha=1, edgecolor='None'
                                 )

    ax.get_legend().remove()
    ax.set_ylabel(' ')
    ax.set_xlabel(' ')

    ax.set_xticks([])
    ax.set_yticks([])

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    fig.savefig('{}/plot_image/2D_Plot.png'.format(output_dir), dpi=main_plot_dpi, bbox_inches='tight')
    plt.close(fig)


    plt.rcParams["font.size"] = gene_plot_font_size
    plt.rcParams['font.family'] = gene_plot_font_family
    plt.ioff()

    for count_idx, (iter_idx, row) in enumerate(tqdm(gene_scores.sort_values('MarcoPolo', ascending=True).iterrows(), total=top_num_image)):
        if count_idx == top_num_image:
            break

        fig = plt.figure(figsize=(3 * 8, 6))
        gs = fig.add_gridspec(6, 3 * 8)
        subplot_list = [fig.add_subplot(gs[0:6, 0:6]),
                        fig.add_subplot(gs[0:6, 6 + 2:6 + 2 + 6])]
        #import ipdb; ipdb.set_trace()
        exp_data_corrected_on = expression_matrix.T[iter_idx][gamma_argmax_list[iter_idx] == 0]
        exp_data_corrected_off = expression_matrix.T[iter_idx][gamma_argmax_list[iter_idx] == 1]

        bins_log = [np.power(1.2, i) for i in range(
            np.max([1, int(np.log(np.max([1, np.max(expression_matrix)])) / np.log(1.2))]))]

        for idx in range(2):
            ax = subplot_list[idx]
            if idx == 0:
                sns.scatterplot(x="coord_x", y="coord_y", hue=plot_value, data=cell_meta_info,
                                palette=get_discrete_palette(
                                    len(plot_value_unique)).tolist() if plot_value.dtype == int else None,
                                ax=ax, alpha=0.3, edgecolor="None",
                                s=gene_plot_s
                                )
                sns.scatterplot(x="coord_x", y="coord_y", data=cell_meta_info.loc[gamma_argmax_list[iter_idx] == 0],
                                ax=ax,
                                edgecolor=[1, 0, 0, 1],
                                facecolors="None",
                                linewidth=1,
                                s=gene_plot_s,
                                zorder=10
                                )
                ax.title.set_text('On-Off in 2D plot')
                ax.get_legend().remove()

                ax.set_xlabel('Dim 1')
                ax.set_ylabel('Dim 2', labelpad=-10)

                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(1.5)

            elif idx == 1:
                ax.title.set_text('Expression of Cells')

                ax.hist(exp_data_corrected_on, bins=bins_log, label='On', color=(1, 0, 0, 0.8))
                ax.hist(exp_data_corrected_off, bins=bins_log, label='Off', color=(0.5, 0.5, 0.5, 0.5))
                ax.set_xscale('log')

                ax.set_ylabel('Cell Counts')
                ax.set_xlabel('Expression count')

                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                leg = ax.legend(loc='upper left',
                                fontsize=15,
                                frameon=False,
                                bbox_to_anchor=(0.22, -0.15),
                                ncol=2,
                                handletextpad=0.2,
                                columnspacing=1.3,
                                markerscale=2.5)
                for rec in leg.get_patches():
                    rec.set_height(8)
                    rec.set_width(15)

                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(2)

        fig.savefig('{}/plot_image/{}.png'.format(output_dir, iter_idx), dpi=gene_plot_dpi, bbox_inches='tight')

        plt.close(fig)


def generate_report(adata: ad.AnnData, size_factor_key: Union[str, None], regression_result: dict, gene_scores: pd.DataFrame, output_dir: str,  low_dim_key:str, cell_color_key=None, gene_info_path: str=None,
                    top_num_html: int=1000, top_num_image: int=1000, mode=2, plot_parameters: dict={}):

    """
    Args:
        adata: anndata.AnnData containing scRNA-seq data. `.X` should be a matrix containing raw count data of shape (# cells, # genes).
        size_factor_key: key of the `adata.obs` containing the size factors. If None, no size factors will be used.
        regression_result: dict containing regression results. Return value of `run_regression` function.
        gene_scores: pd.DataFrame containing gene scores. Return value of `find_markers` function.
        gene_info_path: 'https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz'
        output_dir: directory to save the report
        low_dim_key: key for accessing the 2D coordinates stored in `adata.obsm`
        cell_color_key: key for accessing the variables for coloring cells stored in `adata.obs`. If None, no cell coloring will be used.
        top_num_html: the number of top genes to show in the HTML report. Default: 1000.
        top_num_image: the number of top genes to generate figures. Default: 1000.
        mode: the number of groups used for marker selection. Default: 2.
        plot_parameters: The parameter to be passed to the `generate_image_files` function ex: {"main_plot_s": 25, "main_plot_dpi": 100, "main_plot_font_size": 10, "main_plot_font_family": 'Arial', "gene_plot_s": 10, "gene_plot_dpi": 60, "gene_plot_font_size": 15, "gene_plot_font_family": 'Arial'}

    Returns:

    """
    expression_matrix = adata.X
    num_cells=expression_matrix.shape[0]
    num_genes=expression_matrix.shape[1]

    gene_scores_munge=gene_scores.copy()
    gene_scores_munge['Gene ID']=adata.var.index.values

    output_dir=str(Path(output_dir) / "report")

    ########################
    # Assign cells to on-cells and off-cells
    ########################
    print("Assign cells to on-cells and off-cells...")
    gamma_list = regression_result["gamma_list_cluster"][mode]
    gamma_argmax_list = MarcoPolo.utils.gamma_list_expression_matrix_to_gamma_argmax_list(gamma_list, expression_matrix)

    ########################
    # Calculate voting score
    ########################
    #oncell_size_list = np.sum(gamma_argmax_list == 0, axis=1)
    oncell_size_cliplist = MarcoPolo.utils.gamma_argmax_list_to_oncell_size_list_list(gamma_argmax_list)
    intersection_list = MarcoPolo.utils.gamma_argmax_list_to_intersection_list(gamma_argmax_list)
    intersectioncount_prop=((intersection_list/oncell_size_cliplist))
    intersectioncount_prop_top10=[np.arange(0,len(i))[i>=sorted(i)[-10]][:10] for i in intersectioncount_prop]
    
    gene_scores_munge['Voting_genes_top10']=[gene_scores_munge['Gene ID'][i].values for i in intersectioncount_prop_top10]
    gene_scores_munge_voting=gene_scores_munge.copy()    

    ########################
    # Annotate gene_scores with gene info
    ########################
    if gene_info_path is not None:
        print(f"Annotating genes with the gene info: {gene_info_path}")
        
        gene_info=pd.read_csv(gene_info_path,sep='\t')
        #import ipdb
        #ipdb.set_trace()
        by='ID' if 'ENS' in adata.var.index.values.tolist()[0] else 'name'
        gene_scores_munge=annotate_gene_info(gene_scores=gene_scores_munge, gene_query_list=adata.var.index.values.tolist(), gene_info=gene_info, by=by)

        gene_scores_munge['Log2FC']=(gene_scores_munge['log_fold_change']/np.log10(2)).round(2)

        gene_scores_munge=gene_scores_munge[[
                                    'MarcoPolo_rank',
                                    'Gene ID',
                                    'Symbol', 'description', 'Other_designations', 'type_of_gene',
                                    'Log2FC',
                                    'MarcoPolo',
                                    'bimodality_score_rank',
                                    'voting_score_rank',
                                    'proximity_score_rank',
                                    'oncell_size','oncell_size_rank',
                                    'dbXrefs'
                                   ]]
        gene_scores_munge['img'] = gene_scores_munge.apply(lambda x: '<img src="plot_image/{idx}.png" alt="{idx}">'.format(idx=x.name), axis=1)


    else:
        gene_scores_munge['Log2FC'] = (gene_scores_munge['log_fold_change'] / np.log10(2)).round(2)

        gene_scores_munge=gene_scores_munge[[
                                    'MarcoPolo_rank',
                                    'Gene ID',
                                    'Log2FC',
                                    'MarcoPolo',
                                    'bimodality_score_rank',
                                    'voting_score_rank',
                                    'proximity_score_rank',
                                    'oncell_size','oncell_size_rank',
                                   ]]
        gene_scores_munge['img']=gene_scores_munge.apply(lambda x: '<img src="plot_image/{idx}.png" alt="{idx}">'.format(idx=x.name),axis=1)

    # import ipdb; ipdb.set_trace()
    ########################
    # Generate table files
    ########################
    print(f"Generating table files...")
    generate_html_file(output_dir=output_dir, gene_scores=gene_scores_munge, num_genes=num_genes, num_cells=num_cells, top_num_html=top_num_html)
    gene_scores_munge_voting[['Gene ID','Voting_genes_top10']].to_html('{}/voting_result.html'.format(output_dir))
    gene_scores_munge.to_csv('{}.table.tsv'.format(output_dir), sep='\t')

    ########################
    # Generate image files
    ########################
    print(f"Generating image files...")
    generate_image_files(adata=adata, size_factor_key=size_factor_key, gamma_argmax_list=gamma_argmax_list, gene_scores=gene_scores,
                         low_dim_key=low_dim_key, output_dir=output_dir, top_num_image=top_num_image,
                         cell_color_key=cell_color_key,
                         **plot_parameters)








