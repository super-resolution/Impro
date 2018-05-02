"""
=================================================================
Class to import common microscopy data formats into a numpy array
=================================================================
supported data types are: .czi, .lsm, .tiff, .png

"""
import os
import numpy as np
import xml.etree.ElementTree as XMLET
import tifffile
import pandas as pd
from scipy import misc
from itertools import islice
from libs import czifile
import re

class stormfile(object):
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


class storm_image(object):
    """
    Rewritten class to read STOM/dSTORM data from a text file in a numpy array
    Improved performance and logic
    Can transform the localizations if a matrix is given
    Recives metadata from file header
    """
    def __init__(self, path):
        self.reset_data()
        self.file_path = path

    def reset_data(self):
        self.stormData = []
        self.StormChannelList = []
        self.roi_list = []
        self.coords_cols = []
        self.other_cols = []
        self.matrix = None

    #Read and prepare STORM data
    def parse(self):
        self.isParsingNeeded = False
        localizations = stormfile(self.file_path)
        localizations.readfile()
        localizations.get_header_info()

        #array = stormfile(self.file_path)
        #array.getHeaderInfo()
        self.stormData = localizations.data

        #prevent negative x,y values. Set to Zero
        self.stormData[...,0] = self.stormData[...,0]-self.stormData[...,0].min()
        self.stormData[...,1] = self.stormData[...,1]-self.stormData[...,1].min()
        self.size = self.stormData[...,0].max(), self.stormData[...,1].max()
        #Build structured array with title name and value of columns.
        storm_reshaped = np.negative(np.ones((self.stormData.shape[0], 6)))
        for i,j in enumerate(localizations.dataColumn):
            if j >=0:
                storm_reshaped[...,int(i)] = self.stormData[..., int(j)]
        #set precision to 10 nm if no value given
        if (storm_reshaped[...,2]<0).all():
            storm_reshaped[...,2] = 10
        self.stormData = storm_reshaped



class microscope_image(object):
    """
    :param path: Path to image
    *data* is the image data as numpy array structured as: [Colour, ZStack, X, Y]
    *meta_data* contains all received meta data from file as dict
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
        """
        Load data from file and set meta data
        :param calibration_px: pixel size if no meta data is found
        """
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
        """
        Sets meta_data to px(not recommended)
        :param px: pixel size to set
        """
        self.meta_data['SizeX'] = px
        self.meta_data['SizeY'] = px
        self.meta_data['SizeZ'] = px