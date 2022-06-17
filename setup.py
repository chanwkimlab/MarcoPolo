from setuptools import setup, find_packages

# python -m build
# python -m twine upload --repository pypi dist/* --verbose

setup(
  name = 'marcopolo-pytorch',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '1.0.5',
  description = 'MarcoPolo - Pytorch',
  author = 'Chanwoo Kim',
  author_email = 'ch6845@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/chanwkimlab/MarcoPolo',
  keywords = [
    'single-cell',
    'bioinformatics',
    'pytorch'
  ],
  install_requires=[
    'tqdm',
    'einops>=0.3',
    'numpy>=1.19.2',
    'torch>=1.4.0',
    'pandas>=1.2.0',
    'scikit-learn>=0.24.1',
    'scipy>=1.6.1',
    'matplotlib>=3.3.0',
    'seaborn>=0.11.1',
    'Jinja2>=2.11.2',
    'anndata>=0.7.4',
    'rpy2>=3.4.2',
    'ipywidgets>=7.5.1',
    'scanpy>=1.9.0',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: Other/Proprietary License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)