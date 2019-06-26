from impro.data.image_factory import ImageFactory
from impro.analysis.filter import Filter
from impro.analysis.analysis_facade import *
import unittest
import os


def setting_1():
    #Create and prepare SIM image
    image = ImageFactory.create_image_file(r"D:\asdf\3D Auswertung 22092016\20160919_SIM1\20160919_SIM_0824a_lh_1_Out_Channel Alignment.czi")
    image_array = image.data[:, 8]/2
    image_array = np.clip(image_array[0], 0, 255)
    image_array = (image_array).astype("uint8")[1000:2400, 200:1600]
    image_array = np.flipud(np.fliplr(image_array))
    image_array = np.fliplr(image_array)
    #Create and prepare dSTORM data
    storm = ImageFactory.create_storm_file(r"D:\asdf\3D Auswertung 22092016\20160919_SIM1\20190920_3D_sample0824a_SIM0919_SIM1.txt")
    indices = Filter.local_density_filter(storm.stormData, 100.0, 18)
    storm_data = storm.stormData[indices]

    return image_array,storm_data

def setting_2():
    #Create and prepare SIM image
    image = ImageFactory.create_image_file(r"D:\_3dSarahNeu\!!20170317\20170317_0308c_SIM9_Out_Channel Alignment.czi")
    image_array = image.data[:, 1] / 6
    image_array = np.clip(image_array[0], 0, 255)
    image_array = np.flipud(np.fliplr(image_array))
    image_array = (image_array).astype("uint8")[0:1400, 0:1400]
    image_array = np.fliplr(image_array)
    #Create and prepare dSTORM data
    storm = ImageFactory.create_storm_file(r"D:\_3dSarahNeu\!!20170317\trans_20170317_0308c_SIM9_RG_300_300_Z_coordinates_2016_11_23.txt")
    indices = Filter.local_density_filter(storm.stormData, 100.0, 2)
    storm_data = storm.stormData[indices]

    return image_array,storm_data

def setting_3():
    #Create and prepare SIM image
    image = ImageFactory.create_image_file(r"D:\Microtuboli\Image2b_Al532_ex488_Structured Illumination.czi")
    image_array = image.data[:, 3]/6
    image_array = np.clip(image_array[0], 0, 255)
    image_array = np.flipud(image_array)
    image_array = (image_array).astype("uint8")[0:1400, 0:1400]
    image_array = np.fliplr(image_array)
    #Create and prepare dSTORM data
    storm = ImageFactory.create_storm_file(r"D:\Microtuboli\20151203_sample2_Al532_Tub_1.txt")
    indices = Filter.local_density_filter(storm.stormData, 100.0, 18)
    storm_data = storm.stormData[indices]

    return image_array,storm_data

class TestImageAlignment(unittest.TestCase):
    def setUp(self):
        self.settings = [setting_3(),setting_1(),setting_2()]
    def test_alignment_setting_1(self):
        for i in range(3):
            sim = self.settings[i][0]
            storm_data = self.settings[i][1]
            storm = create_alpha_shape(storm_data, 130)
            col = int(storm.shape[1] / 200)
            row = int(storm.shape[0] / 200)
            self.assertIsNotNone(storm)
            cv2.imshow("Sim", sim)
            print(f"Successfully created STORM image for test data set {i}")
            source_points, target_points, overlay, results = find_mapping(sim, storm, n_row=row, n_col=col)
            source_points, target_points = error_management(results, source_points, target_points, n_row=row)
            try:
                self.assertGreater(len(source_points),7)
            except:
                cv2.imshow("Overlay", overlay.astype(np.uint8))
                cv2.imshow("Sim", sim)
                cv2.imshow("storm", storm)
                cv2.waitKey(0)
                raise ValueError(f"Not enought points for transformation {len(source_points)} in settings {i} found.")

            print(f"Successfully found enought source and target points for transformation")
            M = transform.estimate_transform("affine",source_points,target_points)
            correlation_index = pearson_correlation(sim, cv2.cvtColor(storm, cv2.COLOR_RGBA2GRAY), M)
            self.assertGreater(correlation_index[0], 0.29)

if __name__ == '__main__':
    unittest.main()

