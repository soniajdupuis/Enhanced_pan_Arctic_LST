
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
[<a href=><strong>Training dataset</strong></a>]
[<a href=><strong>Final AVHRR LST dataset</strong></a>]
</p>
  



Code to perform super-resolution on large-scale land surface temperatre (LST) datasets. The present use case downscales a 40-year record of AVHRR LST data (https://zenodo.org/records/13361744) from 0.05° to 0.01° across circumpolar scale.
The algorithm has been adapted from the *Guided Depth Super-Resolution by Deep Anisotropic Diffusion* framework available here: https://github.com/prs-eth/Diffusion-Super-Resolution/tree/main. The main changes are:
- The workflow has been adapted to support geopspatial data, making use of the torchgeo framework
- The guide is now built from digital elevation model (DEM), land cover and capoy height data
- The algortihm has been trained on MODIS LST data availbale from the ESA LST CCI data portal. The extracted scenes used for training, validation and evaluation are available here:

