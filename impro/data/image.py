# Copyright (c) 2018-2022, Sebastian Reinhard
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
:Author:
  `Sebastian Reinhard`

:Organization:
  Biophysics and Biotechnology, Julius-Maximillians-University of Würzburg

:Version: 2019.06.26

"""

import os
import numpy as np
import xml.etree.ElementTree as XMLET
import tifffile
import pandas as pd
from scipy import misc
from itertools import islice
from impro.libs import czifile
from skimage import transform


class StormReader():
    """
    ====================================================
    Find dSTORM columns with regular expressions
    ====================================================
    """
    def __init__(self, input_loc_path):
        self.dataColumn = np.zeros((6))
        self.data = []
        self.size = []
        self.path = input_loc_path

    def readfile(self):
        self.data = pd.read_csv(self.path, skiprows=1, header=None, delim_whitespace=True).as_matrix()

    def clear(self):
        self.data = []
        self.loc_size = []

    def get_header_info(self):
        with open(self.path) as loc:
            self.Header = list(islice(loc, 1))
            headerSplit = self.Header[0].split("/><")
            default_header_args = ["X", "Y", "Precision", "Z", "Emission", "Frame"]
        for j,arg in enumerate(default_header_args):
            for i in range(len(headerSplit)):
                if "semantic" in headerSplit[i]:
                    identifierString = headerSplit[i].split('semantic="')
                    identifier = identifierString[-1].split('"', 1)[0]
                    if arg.lower() in identifier.lower():
                        self.dataColumn[j] = i
                        break
                self.dataColumn[j] = -1
                            # try:
                        #     max_value = headerSplit[i].split('max="')[-1].split('"',1)[0]
                        #     max_value = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", max_value)
                        #     self.dataColumn[arg + " max value"] = float(max_value[0])
                        # except:
                        #     print("no max value" + identifier)



    def save_loc_file(self, path, file):
        x_max = "%.5e" %(file[:,0].max()*10**(-9))
        x_min = 0 #"%.5e" %(loc_input_1[:,0].min()*10**(-9))
        y_max = "%.5e" %(file[:,1].max()*10**(-9))
        y_min = 0 #"%.5e" %(loc_input_1[:,1].min()*10**(-9))
        frame_min = '%u' %file[:2].min()
        header = '<localizations insequence="true" repetitions="variable"><field identifier="Position-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in X" unit="nanometer" min=\"'+str(x_min)+' m\" max=\"'+str(x_max)+' m\" /><field identifier="Position-1-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in Y" unit="nanometer" min=\"'+str(y_min)+' m\" max=\"'+str(y_max)+' m\" /><field identifier="ImageNumber-0-0" syntax="integer" semantic="frame number" unit="frame" min=\"'+str(frame_min)+' fr\" /><field identifier=\"Amplitude-0-0\" syntax=\"floating point with . for decimals and optional scientific e-notation\" semantic=\"emission strength\" unit=\"A/D count\" /><field identifier="FitResidues-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="fit residue chi square value" unit="dimensionless" /><field identifier="Fluorophore-0-0" syntax="integer" semantic="index of fluorophore type" unit="dimensionless" min="0 dimensionless" max="1 dimensionless" /><field identifier="LocalBackground-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="local background" unit="A/D count" /></localizations>'
        np.savetxt(path, file, delimiter=' ', fmt=['%1.1f','%1.1f','%u','%.10g','%.10g','%u','%1.3f'], header= header)


class LocalisationReader():
    """
    ====================================================
    Read RapidSTORM files
    ====================================================

    Rewritten class to read STOM/dSTORM data from a text file in a numpy array
    Improved performance and logic
    Can transform the localizations if a matrix is given
    Recives metadata from file header


    Attributes
    ----------
    path : str
        Path to Rapidstorm .txt file
    stormData: np.array
        Restructured data: X, Y, Precision, Z, Emission, Frame #todo switch Precision and Z
    size: np.array
        X,Y size of data in nanometer


    Methods
    -------
    reset_data()
        resets dSTORM data
    parse()
        read dSTORM data from file
    transform(sklearn.transform)
        Transform dSTORM data with given transformation
    """
    def __init__(self, path):
        self.reset_data()
        self.file_path = path
        self.size = np.array([0,0])

    def reset_data(self):
        self.stormData = []

    #Read and prepare STORM data
    def parse(self):
        self.isParsingNeeded = False
        localizations = StormReader(self.file_path)
        localizations.readfile()
        localizations.get_header_info()

        #array = stormfile(self.file_path)
        #array.getHeaderInfo()
        self.stormData = localizations.data

        #prevent negative x,y values. Set to Zero
        self.stormData[...,0] = self.stormData[...,0]-self.stormData[...,0].min()
        self.stormData[...,1] = self.stormData[...,1]-self.stormData[...,1].min()
        self.size = np.array([self.stormData[...,0].max(), self.stormData[...,1].max()])
        #Build structured array with title name and value of columns.
        storm_reshaped = np.negative(np.ones((self.stormData.shape[0], 6)))
        for i,j in enumerate(localizations.dataColumn):
            if j >=0:
                storm_reshaped[...,int(i)] = self.stormData[..., int(j)]
        #set precision to 10 nm if no value given
        if (storm_reshaped[...,2]<0).all():
            storm_reshaped[...,2] = 10
        self.stormData = storm_reshaped

    def transformAffine(self, path=None, src=None, dst=None):
        if path is not None:
            landmarks = pd.read_csv(path, skiprows=1,engine="c", na_filter=False, header=None, delim_whitespace=True, dtype=np.float32).as_matrix()
            dst = landmarks[:,3:5]
            src = landmarks[:,1:3]
        affine = transform.estimate_transform("affine",src,dst)
        data = self.stormData[0][:,0:2]
        data = affine(data)
        self.stormData[0][:,0:2] = data

class ImageReader():
    """
    ====================================================
    Read Image file
    ====================================================

    Possible formats are .czi, .lsm, .tiff, .png

    Attributes
    ----------
    path : str
        Path to file
    data: np.array
        Image data as array
    metaData: np.array
        Meta Data of image file containing
        computional dimension: ShapeSizeX, ShapeSizeY, ShapeSizeZ, ShapeSizeC
        physical dimension: SizeX(Pixel Size in X direction)...

    Methods
    -------
    reset_data()
        resets data
    parse()
        read data from file
    calibration_px(float)
        Set pixel size manually
    """

    def __init__(self, path):
        self.file_path = path
        self.extend = os.path.splitext(self.file_path)[1]
        self.reset_data()

    #Reset ConfocalImage attributes.
    def reset_data(self):
        self.data = []
        self.meta_data = {}
        self.isParsingNeeded = True

    #Read the image data and metadata und give them into a numpy array.
    #Rearrange the arrays into a consistent shape.
    def parse(self, calibration_px=1.0):
        self.isParsingNeeded = False
        self.meta_data = {}
        self.data = []
        #CZI files
        if self.extend == '.czi':
            with czifile.CziFile(self.file_path) as czi:
                data = czi.asarray()
                Header_Metadata = str(czi).split('<ImageDocument>')
                string = '<ImageDocument>'+Header_Metadata[1]
                #print(string.strip("'"))
                metadata = XMLET.fromstring(string.strip("'"))
                try:
                    #Query XML fore the metadata for picture shape(X;Y;Z-stacks).
                    #Picture Shape.
                    shapes = metadata.findall('./Metadata/Information/Image')[0]
                    self.meta_data["ShapeSizeX"] = int(shapes.findall('SizeX')[0].text)
                    self.meta_data["ShapeSizeY"] = int(shapes.findall('SizeY')[0].text)
                    try:
                        self.meta_data["ShapeSizeZ"] = int(shapes.findall('SizeZ')[0].text)
                    except:
                        self.meta_data["ShapeSizeZ"] = 1
                    #Get the hyperstack dimension if the image is a hyperstack.
                    try:
                        self.meta_data["ShapeSizeC"] = int(shapes.findall('SizeC')[0].text)
                    except:
                        self.meta_data["ShapeSizeC"] = 1
                        print("No info of color channels 1 assumed")
                    #Get physical pixel size of image(nm/px) convert to(µm/px).
                    PixelSizes = metadata.findall('./Metadata/Scaling/Items/Distance')
                    self.meta_data['SizeX'] = float(PixelSizes[0].findall('Value')[0].text)*10**6
                    self.meta_data['SizeY'] = float(PixelSizes[1].findall('Value')[0].text)*10**6
                    self.meta_data['SizeZ'] = float(PixelSizes[2].findall('Value')[0].text)*10**6
                except(ValueError):
                    print ("Metadata fail")

        #Tiff files.
        #Tiff files are problematic because they most likely wont contain the necessary metadata.
        #Try to get the shape info over common dimensions.
        elif self.extend == '.tif':
            with tifffile.TiffFile(self.file_path) as tif:
                data = tif.asarray()
                for shape in data.shape:
                    if shape <5:
                        self.meta_data["ShapeSizeC"] = shape
                    elif shape <40:
                        self.meta_data["ShapeSizeZ"] = shape
                    else:
                        self.meta_data["ShapeSizeY"] = shape
                        self.meta_data["ShapeSizeX"] = shape

        #Read Lsm Files.
        elif self.extend == '.lsm':
            with tifffile.TiffFile(self.file_path) as tif:
                data = tif.asarray(memmap=True)
                headerMetadata = str(tif.pages[0].cz_lsm_scan_info)
                metadataList = headerMetadata.split("\n*")
                #Get image shape from lsm header SizeC=0 if not given.
                for shapes in metadataList:
                    if "images_height" in shapes:
                        self.meta_data["ShapeSizeX"]= int(shapes.split()[-1])
                    if "images_width" in shapes:
                        self.meta_data["ShapeSizeY"]= int(shapes.split()[-1])
                    if "images_number_planes" in shapes:
                        self.meta_data["ShapeSizeZ"]= int(shapes.split()[-1])
                    if "images_number_channels" in shapes:
                        self.meta_data["ShapeSizeC"]= int(shapes.split()[-1])
                #Get physical pixel size of image(nm/px) convert to(µm/px).
                data = np.swapaxes(data,1,2)
                lsm_header = str(tif.pages[0].tags.cz_lsm_info)
                LsmInfo = lsm_header.split(", ")
                i = 0
                #Query for pixel size.
                for element in LsmInfo:
                    if "e-0" in element:
                        i += 1
                        if i == 1:
                            self.meta_data['SizeX'] = (float(element)*10**6)
                        if i == 2:
                            self.meta_data['SizeY'] = (float(element)*10**6)
                        if i == 3:
                            self.meta_data['SizeZ'] = (float(element)*10**6)

        elif self.extend == ".png":
            data = misc.imread(self.file_path)
            data = np.expand_dims(np.expand_dims(data[...,0],0),0)
            self.meta_data["ShapeSizeC"] = 1
            self.meta_data["ShapeSizeZ"] = 1
            self.meta_data["ShapeSizeX"] = data.shape[2]
            self.meta_data["ShapeSizeY"] = data.shape[3]
            self.meta_data["SizeZ"] = 1
            self.meta_data["SizeX"] = 0.01
            self.meta_data["SizeY"] = 0.01
        #Bring all formats in the same shape.
        self.data = np.reshape(data,(self.meta_data["ShapeSizeC"],self.meta_data["ShapeSizeZ"],self.meta_data["ShapeSizeX"],self.meta_data["ShapeSizeY"]))
        self.meta_data['ChannelNum'] = self.meta_data["ShapeSizeC"]
        #Set pixel size to manuell value if there are no metadata.
        if self.meta_data == {}:
            self.set_calibration(calibration_px)
        #Set the box for manuel calibration to the actuell pixel size.

    #Set pixel size to manuell value.
    def set_calibration(self, px):
        self.meta_data['SizeX'] = px
        self.meta_data['SizeY'] = px
        self.meta_data['SizeZ'] = px