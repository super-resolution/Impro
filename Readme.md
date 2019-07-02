# Impro
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2620224.svg)](https://doi.org/10.5281/zenodo.2620224)
[![Documentation Status](https://readthedocs.org/projects/impro/badge/?version=latest)](https://impro.readthedocs.io/en/latest/?badge=latest)

Impro is a package for data processing in super-resolution microscopy. It contains high perfomant
GPU based visualization and algorithms for 2D and 3D data. <br />
Main features:
* Cuda accelerated 2D Alpha Shapes
* Automated image alignment via General Hough Transform
* Huge list of filters for localization datasets
* Customizable Opengl widget based on modelview perspective standard
* Pearson correlation between alpha shape and image data

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

* Nvidia GPU with compute capability 3 or higher
* Cuda 9.0 from [CUDA website](https://developer.nvidia.com/cuda-90-download-archive) (CUDA 10 is yet not supported by pycuda)

```
Some images
```

### Installing

1. Clone git repository
2. Open cmd and cd to repository
3. Install requirements with:
```
pip install -r requirements.txt
```
4. Run:
```
python setup.py install
```

### Examples
The file
```
impro/main.py
```
provides example code for using the package. An Impro based GUI can be found at [super-resolution correlator](https://github.com/super-resolution/Super-resolution-correlator)
