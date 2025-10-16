# PIGEON-FEATHER


![method](./docs/image/pigeon_feather.png)

PIGEON-FEATHER (*Peptide ID Generation for Exchange Of Nuclei-Free Energy Assignment Through Hydrogen Exchange Rates*): a method for calculating free energies of opening (∆Gop) at single- or near-single-amino acid resolution for protein ensembles of all sizes from hydrogen exchange/mass spectrometry (HX/MS) data. 


## Documentation

PIGEON-FEATHER can be used from Jupyter Notebook (recommended), or by writing Python scripts. The documentation can be found at [http://glasgowlab.org/PIGEON-FEATHER](http://glasgowlab.org/PIGEON-FEATHER/).

The tutorials notebooks could be found at docs/source/tutorials.

## Compatibiliy with HXMS format 

[HXMS](https://www.biorxiv.org/content/10.1101/2025.10.14.682397v1) is a unified, lightweight, and human-readable file format that preserves full isotopic envelopes and experimental conditions for HX/MS data. We recommend using HXMS with FEATHER. HXMS files can be generated using [PFLink](https://huggingface.co/spaces/glasgow-lab/PFLink), which supports exports from BioPharma Finder, HDExaminer, DynamX, and HDX Workbench.  

You can easily read and write HXMS files using **FEATHER**:
```python
from pigeon_feather.hxio import load_HXMS_files, write_HXMS_files

hdxms_data = load_HXMS_files(["./ecDHFR_APO.hxms",
                              "./ecDHFR_TRI.hxms"])

write_HXMS_files([hdxms_data], protein_name="ecDHFR", output_dir=".")
```


## Installation

```
# install mamba
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh 

# create a conda environment and install the required packages
conda create --name pigeon_feather python=3.11
conda activate pigeon_feather

conda install jupyterlab mdanalysis numba
conda install pymol-open-source

pip install pyopenms hdxrate

# clone the PIGEON-FEATHER repo
git clone https://github.com/glasgowlab/PIGEON-FEATHER.git

# clone the bayesian hdx iso repo
git clone https://github.com/lucl13/bayesian_hdx.git

cd PIGEON-FEATHER
pip install .

cd ../bayesian_hdx
pip install .
```

## Docker
```
docker build -f docker/Dockerfile -t pigeon_feather:0.9 .
docker run -it -v $(pwd):/work -p 8889:8889 --rm pigeon_feather:0.9 jupyter-lab --port 8889
```

click the link in the terminal to open the jupyter notebook
cd to example/docker_example
click and run


> **NOTE:** bayesian_hdx at the above link is a fork of the original repo at https://github.com/salilab/bayesian_hdx with some modifications to the code to make it compatible with the PIGEON-FEATHER and support for the isotopic envelope fitting.


## Citation
If you use PIGEON-FEATHER in scientific work, please cite:

> C. Lu*, M. L. Wells*, A. Reckers, and A. Glasgow, “Site-resolved energetic information from HX/MS experiments.” bioRxiv, 2024. (*equal contributions) [link to preprint](https://www.biorxiv.org/content/10.1101/2024.08.04.606547v1)
