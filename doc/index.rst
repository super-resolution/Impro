.. impro documentation master file, created by
   sphinx-quickstart on Thu Jun 27 16:06:18 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to impro's documentation!
=================================
`Impro <https://github.com/super-resolution/impro>`_. is a package for data processing in super-resolution microscopy. It contains high perfomant
GPU based visualization and algorithms for 2D and 3D data.
Main features:

 * Cuda accelerated 2D Alpha Shapes
 * Automated image alignment via General Hough Transform
 * Huge list of filters for localization datasets
 * Customizable Opengl widget based on modelview perspective standard
 * Pearson correlation

References
----------
A detailed explantation of the algorithm and it's functionality can be found at:

`Reinhard S, Aufmkolk S, Sauer M, Doose S. 
Registration and Visualization of Correlative Super-Resolution Microscopy Data. 
Biophys J. 2019 Jun; 116(11) 2073-2078. 
doi:10.1016/j.bpj.2019.04.029. PMID: 31103233. <https://doi.org/10.1016/j.bpj.2019.04.029>`_

Example
-------
Basic usage example. Preprocessing is adapted to the dataset provided in 
`Super-resolution correlator <https://github.com/super-resolution/Super-resolution-correlator>`_.

.. code-block:: python
   
   from impro.data.image_factory import ImageFactory
   from impro.analysis.filter import Filter
   from impro.analysis.analysis_facade import *
   
   # Read and preprocess SIM image
   image = ImageFactory.create_image_file(r"path_to_file.czi")
   
   # Example for image preprocessing
   image_array = image.data[:, 3] / 6
   image_array = np.clip(image_array[0], 0, 255)
   image_array = np.flipud(image_array)
   image_array = (image_array).astype("uint8")[0:1400, 0:1400]
   image_array = np.fliplr(image_array)
   
   # Read dSTORM data
   storm = ImageFactory.create_storm_file(r"path_to_file.txt")
   
   # Preprocess dSTORM point data
   indices = Filter.local_density_filter(storm.stormData, 100.0, 18)
   storm_data = storm.stormData[indices]
   # Render dSTORM data to image
   im = create_alpha_shape(storm_data, 130)
   col = int(im.shape[1]/200)
   row = int(im.shape[0]/200)
   source_points, target_points, overlay, results = find_mapping(sim, storm, n_row=row, n_col=col)
   source_points, target_points = error_management(results, source_points, target_points, n_row=row)
   M = transform.estimate_transform("affine",source_points,target_points)
   correlation_index = pearson_correlation(sim, cv2.cvtColor(storm, cv2.COLOR_RGBA2GRAY), M)

.. toctree::
   :maxdepth: 2

   impro.analysis
   impro.data
   impro.render
   impro.render.common
   modules
   impro.setup
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
