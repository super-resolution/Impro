# Impro

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
* Cuda 8.0 or higher
* pycuda
* Python opencv

```
Some images
```

### Installing

1. Clone git repository
2. Open cmd and cd to repository

```
python setup.py
```
3. Install missing packages if necessary

### Examples
The file
```
impro/main.py
```
provides example code for using the package. An Impro based GUI can be found at [super-resolution correlator](https://github.com/super-resolution/Super-resolution-correlator)
