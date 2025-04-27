Installation
============

Conda
-----

.. code-block:: bash

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



Docker
------

.. code-block:: bash

    docker build -f docker/Dockerfile -t pigeon_feather:0.9 .
    docker run -it -v $(pwd):/work -p 8889:8889 --rm pigeon_feather:0.9 jupyter-lab --port 8889

To open the Jupyter notebook, please click the link displayed in the terminal. 
For users operating on Apple Silicon machines, it may be necessary to append 
`--platform amd64` when building the image, as `pyopenms` is not supported on 
the linux/arm64 architecture. It is advisable to avoid running the Docker container 
on Apple Silicon machines due to the significant performance degradation 
caused by the emulation of the x86_64 architecture.

Note: ``bayesian_hdx`` at the above link is a fork of the `original repo <https://github.com/salilab/bayesian_hdx>`_ , 
which has been modified for compatibility with the PIGEON-FEATHER workflow and support for isotopic envelope fitting.