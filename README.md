

![MarcoPolo](assets/mp.png)

MarcoPolo is a method to discover differentially expressed genes in single-cell RNA-seq data without depending on prior clustering



## Overview


<img src="assets/overview.png" width="700">


`MarcoPolo` is a novel clustering-independent approach to identifying DEGs in scRNA-seq data. MarcoPolo identifies informative DEGs without depending on prior clustering, and therefore is robust to uncertainties from clustering or cell type assignment. Since DEGs are identified independent of clustering, one can utilize them to detect subtypes of a cell population that are not detected by the standard clustering, or one can utilize them to augment HVG methods to improve clustering. An advantage of our method is that it automatically learns which cells are expressed and which are not by fitting the bimodal distribution. Additionally, our framework provides analysis results in the form of an HTML file so that researchers can conveniently visualize and interpret the results.


|Datasets|URL|
|:---|:---|
|Human liver cells (MacParland et al.)|[https://chanwkimlab.github.io/MarcoPolo/HumanLiver/](https://chanwkimlab.github.io/MarcoPolo/HumanLiver/)|
|Human embryonic stem cells (The Koh et al.)|[https://chanwkimlab.github.io/MarcoPolo/hESC/](https://chanwkimlab.github.io/MarcoPolo/hESC/)|
|Peripheral blood mononuclear cells (Zheng et al.)|[https://chanwkimlab.github.io/MarcoPolo/Zhengmix8eq/](https://chanwkimlab.github.io/MarcoPolo/Zhengmix8eq/)|


## Dataset preparation
MarcoPolo works jointly with [AnnData](https://anndata.readthedocs.io/), a flexible and efficient data format for scRNA-seq data widely used in python community. This enables MarcoPolo to seamlessly work with other popular single cell software packages such as [scanpy](https://scanpy.readthedocs.io/), or more broadly, other packages included in the [scverse](https://scverse.org/projects/) project, etc. 

You should prepare your scRNA-seq data in AnnData object before running MarcoPolo.
Please refer to the [AnnData-Getting started](https://anndata-tutorials.readthedocs.io/en/latest/getting-started.html) to get to know more about AnnData.
If your data is in seurat object, you can very easily convert it to AnnData following the instructions [here](https://satijalab.org/seurat/articles/conversion_vignette.html).

As MarcoPolo runs on raw count data, anndata should contain the raw count data in `.X`. The structure of Anndata is described [here](https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.html). 

## Running MarcoPolo on Google Colab 
The easiest way to use MarcoPolo is to use Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chanwkimlab/MarcoPolo/blob/main/notebooks/MarcoPolo.ipynb)

Google colab is a free cloud-based environment for running Python code. Colab allows you to execute MarcoPolo in your browser without any configurations and GPU resources on your end.

- Note
  - User can specify the number of genes to include in the report file by setting the `top_num_table`
    and `top_num_figure` parameters.
  - If there are any two genes with the same MarcoPolo score, a gene with a larger fold change value is prioritized.

The function outputs the two files:

- report/hESC/index.html (MarcoPolo HTML report)
- report/hESC/voting.html (For each gene, this file shows the top 10 genes of which on/off information is similar to the
  gene.)

## Running MarcoPolo on your local machine
### How to install MarcoPolo
We recommend using the following pipeline to install MarcoPolo.
1. Anaconda

Please refer to https://docs.anaconda.com/anaconda/install/linux/ to install Anaconda.
Then, please make a new conda environment and activate it.
```
conda create -n MarcoPolo python=3.8
conda activate MarcoPolo
```

2. Pytorch

Please install `PyTorch` from https://pytorch.org/ (If you want to install CUDA-supported PyTorch, please install CUDA in advance)

3. MarcoPolo

You can simply install MarcoPolo by using the `pip` command:
```bash
pip install marcopolo-pytorch
```
### How to get an updated version of MarcoPolo
You can simply update MarcoPolo by using the `pip` command:
```bash
pip install marcopolo-pytorch --upgrade
```

## Citation

If you use any part of this code or our data, please cite our
[paper](https://doi.org/10.1093/nar/gkac216).

```
@article{kim2022marcopolo,
  title={MarcoPolo: a method to discover differentially expressed genes in single-cell RNA-seq data without depending on prior clustering},
  author={Kim, Chanwoo and Lee, Hanbin and Jeong, Juhee and Jung, Keehoon and Han, Buhm},
  journal={Nucleic Acids Research},
  year={2022}
}
```

## Contact
If you have any inquiries, please feel free to contact
- [Chanwoo Kim](https://chanwoo.kim) (Paul G. Allen School of Computer Science & Engineering @ the University of
  Washington)