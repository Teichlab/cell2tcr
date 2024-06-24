<p align="left"><img src="https://user-images.githubusercontent.com/22446690/230076528-655a571b-6516-4315-b310-36e0d43cfe31.png" width="400"></p>

Cell2TCR is a tool for inference of T cell receptor (TCR) motifs. A TCR motif describes a group of TCRs with sufficient sequence similarity to likely recognise a common epitope.

## Getting started

See example use in [cell2tcr.ipynb](cell2tcr.ipynb).

## Installation

1. Create a new conda environment

```
conda create --name cell2tcr_env python=3.10
conda activate cell2tcr_env
```

2. Install Cell2TCR from Github

```bash
git clone https://github.com/Teichlab/cell2tcr.git
cd cell2tcr
pip install .
```

3. Optional: Add kernel for use in Jupyter notebooks

```
python -m ipykernel install --user --name cell2tcr_env
```

## Human SARS-CoV-2 challenge uncovers local and systemic response dynamics
[Published in Nature 19.06.2024](https://doi.org/10.1038/s41586-024-07575-x)

Rik G. H. Lindeboom*, Kaylee B. Worlock*, Lisa M. Dratva, Masahiro Yoshida, David Scobie, Helen R. Wagstaffe, Laura Richardson, Anna Wilbrey-Clark, Josephine L. Barnes, Lorenz Kretschmer, Krzysztof Polanski, Jessica Allen-Hyttinen, Puja Mehta, Dinithi Sumanaweera, Jacqueline M. Boccacino, Waradon Sungnak, Rasa Elmentaite, Ni Huang, Lira Mamanova, Rakesh Kapuge, Liam Bolt, Elena Prigmore, Ben Killingley, Mariya Kalinova, Maria Mayer, Alison Boyers, Alex Mann, Leo Swadling, Maximillian N. J. Woodall, Samuel Ellis, Claire M. Smith, Vitor H. Teixeira, Sam M. Janes, Rachel C. Chambers, Muzlifah Haniffa, Andrew Catchpole, Robert Heyderman, Mahdad Noursadeghi, Benny Chain, Andreas Mayer, Kerstin B. Meyer, Christopher Chiu, Marko Z. Nikolić† & Sarah A. Teichmann†
