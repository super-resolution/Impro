"""

"""
from impro.data.image import MicroscopeImage as image
from impro.data.image import StormImage as storm
from impro.analysis import filter
from impro.analysis.analysis_facade import *
import os

def setting_1():
    i = image(r"D:\asdf\3D Auswertung 22092016\20160919_SIM1\20160919_SIM_0824a_lh_1_Out_Channel Alignment.czi")
    i.parse()
    y = i.data[:, 8]/2
    y = np.clip(y[0], 0, 255)
    data = storm(r"D:\asdf\3D Auswertung 22092016\20160919_SIM1\20190920_3D_sample0824a_SIM0919_SIM1.txt")
    data.parse()
    storm_data = filter.local_density_filter(data.stormData, 100.0, 18)
    y = (y).astype("uint8")[1000:2400, 200:1600]
    y = np.flipud(np.fliplr(y))
    print(data.stormData[...,3].max())

    return y,storm_data

def setting_2():
    i = image(r"D:\_3dSarahNeu\!!20170317\20170317_0308c_SIM9_Out_Channel Alignment.czi")
    i.parse()
    y = i.data[:, 1] / 8
    y = np.clip(y[0], 0, 255)
    y = np.flipud(np.fliplr(y))
    y = (y).astype("uint8")[0:1400, 0:1400]
    data = storm(r"D:\_3dSarahNeu\!!20170317\trans_20170317_0308c_SIM9_RG_300_300_Z_coordinates_2016_11_23.txt")
    data.parse()
    storm_data = filter.local_density_filter(data.stormData, 100.0, 2)
    return y,storm_data

def setting_3():
    i = image(r"D:\Microtuboli\Image2b_Al532_ex488_Structured Illumination.czi")
    i.parse()
    y = i.data[:, 3]/6
    y = np.clip(y[0], 0, 255)
    y = np.flipud(y)
    y = (y).astype("uint8")[0:1400, 0:1400]
    data = storm(r"D:\Microtuboli\20151203_sample2_Al532_Tub_1.txt")
    data.parse()
    storm_data = filter.local_density_filter(data.stormData, 100.0, 18)
    return y,storm_data
#i = image(r"D:\Microtuboli\Image2b_Al532_ex488_Structured Illumination.czi")

#y = i.data[:,0]/2

# i = image(r"D:\_3dSarahNeu\!!20170317\20170317_0308c_SIM10_Out_Channel Alignment.czi")
#data = storm(r"D:\Microtuboli\20151203_sample2_Al532_Tub_1.txt")


# todo: photon filter, z filter
# x = Filter.photon_filter(data.stormData,3000,99999)

y,storm_data = setting_3()


# import matplotlib.pyplot as plt
# from scipy.spatial import Voronoi, voronoi_plot_2d
# vor = Voronoi(storm_data[...,0:2])
# regions, vertices = voronoi_finite_polygons_2d(vor, radius=130)
#
# for region in regions:
#     polygon = vertices[region]
#     plt.fill(*zip(*polygon), alpha=0.4)
#
# #plt.plot(points[:,0], points[:,1], 'ko')
# plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
# plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
#
# plt.show()
# def find_optimal_alpha():
#     bound = [0,5000,41500,43689]
#     data = storm_data[np.where((storm_data[...,0]>bound[0])&(storm_data[...,0]<bound[1])&(storm_data[...,1]>bound[2])&(storm_data[...,1]<bound[3]))]
#     #data = data[np.where(data[...,0]<5000)]
#     im = create_storm(data)
#     cv2.imshow("asdf",im)
#     cv2.waitKey(0)
#
#     points=[]
#     for p in data:
#         points.append(Point_2(np.int32(p[0]),np.int32(p[1])))
#     print("Created_points")
#     alpha = Alpha_shape_2(10,points)
#     print("alpha =",np.sqrt(alpha.find_optimal_alpha(np.int64(1))))

#find_optimal_alpha()

im = create_alpha_shape(storm_data, 130)

#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\test\storm.jpg",storm_image)
#cv2.imshow("a",y.astype("uint8"))
#cv2.imshow("asdf",im.astype("uint8"))
#cv2.waitKey(0)
#buffer image
#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\test\alpha2.jpg",im)

#im = cv2.imread(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\test_image.tif",-1)

#check alpha shape


#todo:resize images
#y =np.flipud(y).astype(np.uint8)
#y = (y).astype("uint8")[0:1700,0:1700]

#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\test\SIM2.jpg",y)
#cv2.imshow("microSIM", y)
#cv2.waitKey(0)


col = int(im.shape[1]/200)
row = int(im.shape[0]/200)
#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\weighting\image.png",im.astype(np.uint8))
#cv2.waitKey(0)
#norm_source = np.linalg.norm(cv2.cvtColor(im,cv2.COLOR_RGBA2GRAY))
#norm_target = np.linalg.norm(y)
#y = (y.astype(np.float32) * (norm_source/norm_target))
#y = np.clip(y, 0, 255)


points1,points2,z,t_list = find_mapping(y.astype(np.uint8), im,n_row=row,n_col=col)
#print(z.max())
#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\weighting\not_weighted.png",z.astype(np.uint8))
#cv2.imshow("sdgag", z.astype(np.uint8))
#cv2.imshow("y",y)
#cv2.waitKey(0)
#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\Hough_complete_.jpg",z)


p1, p2 = error_management(t_list, points1, points2,n_row = row)
M = transform.estimate_transform("affine",p1,p2)

def test_transformation_accuracy(offset, source_image):
    point1, point2, z, t_lis = find_mapping(y.astype(np.uint8), source_image, n_row=5, n_col=5, offset=offset)
    p1, p2 = error_management(t_lis, point1, point2, n_row=5)
    for coord in p2:
        for i in range(10):
            for j in range(10):
                z[int(coord[1]+i),int(coord[0]+j)]=np.array([1,0,0,1])*255
    T =transform.estimate_transform("affine", p1, p2)

    mask = np.zeros_like(y)
    mask[0:source_image.shape[0], 0:source_image.shape[1]] = 1
    asdf = np.zeros_like(y)
    asdf[0:source_image.shape[0], 0:source_image.shape[1]] = cv2.cvtColor(source_image, cv2.COLOR_RGBA2GRAY)
    coeff = test_pearson(y, asdf, mask, T)[0]

    return T,z,coeff

def evaluate():
    for j in range(20):
        #try:
            alpha = 20+10*j
            im = create_alpha_shape(storm_data, alpha)
            translation  = []
            for i in range(10):
                try:
                    M,z,coeff = test_transformation_accuracy(i*10, im)
                    translation.append(np.array([M.translation[0],M.translation[1],M.rotation,M.shear, coeff]))
                except Exception as error:
                    print('Caught this error: ' + repr(error))
            z[z>255] = 0
            results = np.asarray(translation)
            np.savetxt(os.getcwd()+r"\test_files\results_3\alpha_"+str(alpha)+"txt", results)
            mean_x = np.mean(results[...,0])
            print("error_x",np.mean(results[...,0]),np.mean(np.abs(results[...,0]-mean_x)),"std", np.std(results[...,0]))
            mean_y = np.mean(results[...,1])
            print("error_y",np.mean(results[...,1]), np.mean(np.abs(results[...,1]-mean_y)),"std", np.std(results[...,1]))
            print("error_rot",np.mean(results[...,2]),"std", np.std(results[...,2]))
            print("error_shear",np.mean(results[...,3]),"std", np.std(results[...,3]))
        #except:
        #    print("fail...")
    return M,z,im


#M,z,im = evaluate()



pearsonRGB = cv2.cvtColor(y, cv2.COLOR_GRAY2RGB)
mask = np.zeros_like(y)
mask[0:im.shape[0], 0:im.shape[1]] = 1
asdf = np.zeros_like(y)
asdf[0:im.shape[0], 0:im.shape[1]] = cv2.cvtColor(im, cv2.COLOR_RGBA2GRAY)
test_pearson(y, asdf ,mask , M)
color_warp = np.zeros_like(pearsonRGB)
color_warp[0:im.shape[0], 0:im.shape[1]] = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
dst = transform.warp(color_warp,inverse_map=M.inverse)*255
dst = dst
cv2.imshow("res", dst)
added = dst.astype(np.uint16) + pearsonRGB.astype(np.uint16)
added[added>255] = 0
cv2.imshow("added", added.astype(np.uint8))
#cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\aligned\raw\microtub.tif",added.astype(np.uint8))


cv2.imshow("sdgag", z.astype(np.uint8))# img.astype("uint8"))

cv2.waitKey(0)

#test data1
#Pearson :0.324,ld100;18;alpha130
#test data2
#slice 0; ld100;5;alpha 100
#test data3
#slice 4; ld100; 18; alpha 130

#4541111115632106