<p align="left"><img src="https://user-images.githubusercontent.com/22446690/230076528-655a571b-6516-4315-b310-36e0d43cfe31.png" width="400"></p>

Cell2TCR is a tool for inference of T cell receptor (TCR) motifs. A TCR motif describes a group of TCRs with sufficient sequence similarity to likely recognise a common epitope.

## Installation

1. Create a new conda environment

```
conda create --name cell2tcr_env
conda activate cell2tcr_env
```

2. Install Cell2TCR from Github
Check out the branch ```db_extension``` to perform integrated IEDB.org queries. 

```bash
git clone https://github.com/Teichlab/cell2tcr.git
cd cell2tcr
git fetch origin
git checkout db_extension
pip install .
```

3. Optional: Add kernel for use in Jupyter notebooks

```
python -m ipykernel install --user --name cell2tcr_env
```

## Usage
```
import cell2tcr

# infer motifs
cell2tcr.motifs(df)

# plot largest motif <-> motif 0
cell2tcr.draw_cdr3(df[df.motif==0])

# get all TCR-beta chain matches with IEDB.org database
scores = cell2tcr.db_match(df['IR_VDJ_1_junction_aa'].values)

# annotate original df
cell2tcr.db_annotate(df, scores, 'IR_VDJ_1_junction_aa')

```


### Tutorial on a COVID-19 dataset

See complete example in [cell2tcr.ipynb](cell2tcr.ipynb).

