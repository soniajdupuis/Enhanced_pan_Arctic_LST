
<p align="center">
<h2 align="center"> Guided Super-Resolution for land surface temperature datasets </h2>

<p align="center">
    Sonia Dupuis, Nando Metzger, Konrad Schindler, Frank Goettsche, and Stefan
Wunderle,
    
</p>

<p align="center">
sonia.dupuis@unibe.ch

<p align="center">
[<a href=><strong>Paper</strong></a>]
[<a href="https://zenodo.org/records/17341544"><strong>Training dataset</strong></a>]
[<a href="https://boris-portal.unibe.ch/entities/product/761f8e2f-fb77-4efc-beaf-d196c000ffea"><strong>AVHRR LST dataset</strong></a>]
</p>
  



Code to perform super-resolution on large-scale land surface temperatre (LST) datasets. The present use case downscales a 40-year record of AVHRR LST data (https://zenodo.org/records/13361744) from 0.05° to 0.01° across circumpolar scale.
The algorithm has been adapted from the *Guided Depth Super-Resolution by Deep Anisotropic Diffusion* framework available here: https://github.com/prs-eth/Diffusion-Super-Resolution/tree/main. The main changes are:
- The workflow has been adapted to support geopspatial data, making use of the torchgeo framework
- The guide is now built from digital elevation model (DEM), land cover and capoy height data
- The algortihm has been trained on MODIS LST data availbale from the ESA LST CCI data portal. The extracted scenes used for training, validation and evaluation are available here:


## Installation
The framework relies primarily on Torchgeo and Pytorch. The package necessary to access the ESA CCI data with the ESA CCI toolbox (https://climate.esa.int/en/data/toolbox/) are also included. The necessary librairies with all required dependencies can be installed by running
```bash
conda env create -f environment.yaml
conda activate pytorch_pip
```
### Data

The MODIS LST data used for training, validation and evaluation can be found here:

https://zenodo.org/records/17341544

The coarse GAC data are available here, under the variable 'LST-GAC':

https://boris-portal.unibe.ch/entities/product/761f8e2f-fb77-4efc-beaf-d196c000ffea
## Training

Run the training script via
```bash
python run_train.py --dataset <...> --data-dir <...> --save-dir <...>
```

## Evaluation

For test set evaluation, run

```bash
python run_eval.py --checkpoint <...> --dataset <...> --data-dir <...>
```


## Citation
```
EarthArxiv
```
