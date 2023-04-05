<p align="left"><img src="https://user-images.githubusercontent.com/22446690/230076528-655a571b-6516-4315-b310-36e0d43cfe31.png" width="400"></p>

Cell2TCR is a tool for inference of T cell receptor (TCR) motifs. A TCR motif describes a group of TCRs with sufficient sequence similarity to likely recognise a common epitope.

## Getting started

See example use in [cell2tcr.ipynb](cell2tcr.ipynb).

## Installation

1. Create a new conda environment

```
conda create --name cell2tcr_env
conda activate cell2tcr_env
```

2. Install Cell2TCR from Github

```bash
git clone git@github.com/teichlab/cell2tcr.git
cd cell2tcr
pip install .
```

3. Optional: Add kernel for use in Jupyter notebooks

```
python -m ipykernel install --user --name cell2tcr_env
```
