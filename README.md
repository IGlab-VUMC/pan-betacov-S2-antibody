# pan-betacov-S2-antibody
# Repo for "Discovery and Characterization of a Pan-betacoronavirus S2-binding antibody".
[Paper](https://pubmed.ncbi.nlm.nih.gov/38293237/)

Nicole V. Johnson*, Steven C. Wall*, Kevin J. Kramer*, Clinton M. Holt*, Sivakumar Periasamy, Simone Richardson, Naveenchandra Suryadevara, Emanuele Andreano, Ida Paciello, Giulio Pierleoni, Giulia Piccini, Ying Huang, Pan Ge, James D. Allen, Naoko Uno, Andrea Shiakolas, Kelsey Pilewski, Rachel S. Nargi, Rachel E. Sutton, Alexandria A. Abu-Shmais, Robert Parks, Barton Haynes, Robert H. Carnahan, James E. Crowe Jr., Emanuele Montomoli, Rino Rappuoli, Alexander Bukreyev, Ted M. Ross, Giuseppe A. Sautto#, Jason S. McLellan#, Ivelin S. Georgiev#

*These authors contributed equally.
#Corresponding authors. Email: ivelin.georgiev@vanderbilt.edu, jmclellan@austin.utexas.edu, sauttog@ccf.org.

Clinton Holt at clinton.m.holt@vanderbilt.edu is the repo author.

# Brief Synopsis of the paper:
- B cells were characterized from two donors by LIBRA seq
- The first donor (corresponding to samples 54041, 54042, and 54043) had previously been infected with SARS-CoV-2 and all high affinity cross-reactive mAbs originated from this donor.
- The second donor (corresponding to sample 54044) had no known history of SARS-CoV-2 infection.
- 20/50 characterized mAbs cross-react between SARS1 & SARS2
- 5/50 characterized mAbs bind at least 3 Human CoVs
- mAb 54043-5 is particularly interesting. It is non-neutralizing, but bound all spikes tested from the A, B, and C lineages of Betacoronaviruses as well as in the subgenuses Alphacoronavirus/Pedacovirus and Deltacoronavirus/Buldecovirus.  
- 54043-5 with Fc mutations meant to silence Fc effector functions (LALA-PG) resulted in protection for both prophylactic and therapeutic treatment of mice against SARS-CoV-2 infection.

# Recreating paper figures
- follow along in the `prep_paper_figures.ipynb` jupyter notebook

# Setup Instructions
- The packages are pretty simple overall (except for downloading the mkdssp executable if you want to run that section)
- You could just go through the jupyter notebook and uncomment the lines meant to download packages (to do this I'd recommend python 3.8) otherwise you can use the environment.yml or requirements.txt files as described below.

## Using Conda (Recommended)
1. **Clone the repository**
```bash
git clone https://github.com/IGlab-VUMC/pan-betacov-S2-antibody.git
cd pan-betacov-S2-antibody
```

2. **Create the Conda environment:**
```bash
conda env create -f environment.yml
conda activate panbetacov
```

## Using requirements.txt
1. **Clone the repository:**
```bash
git clone https://github.com/IGlab-VUMC/pan-betacov-S2-antibody.git
cd pan-betacov-S2-antibody
```
2. **Install dependencies:**
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```
