# Super Resolution modeling of PySM simulations

In this repository we show our progress in the training and modeling the [SR3 super resolution model](https://github.com/javierhndev/Super-Resolution-SR3) with PySM generated images. 

## PySM simulations and analysis
In the Notebook `generate_galactic_dust_realization` we do some exploration of [PySM](https://pysm3.readthedocs.io/en/latest/) simulations, [pixell](https://pixell.readthedocs.io/en/latest/) (to convert sky map to 2D), pillow usage, saving figures...

For the last version of the Notebook, check the date in front of the file as `YYYYMM_thenotebook`

## Dataset generation
The `dataset_generation.py` is a Python script that generates images from PySM simulations in 2D.

## Profiling
The performance of the SR3 model on the Voyager HPC system has been analyzed. Check [that section](20250905_voyager_profiling) for more details.
