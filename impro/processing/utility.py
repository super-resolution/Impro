import cv2
import numpy as np
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from ..visualisation import Render as rend
from ..processing import CudaAlphaShape as alpha
from ..processing.CudaHough import hough_transform
from skimage import transform
from scipy.stats.mstats import pearsonr
from PyQt5.QtCore import QPoint


app = pg.mkQApp()



def create_alpha_shape(storm_data, alpha_value):
    k_simplices = alpha.get_k_simplices(storm_data[...,0:2])[0]

    points = np.empty((2*k_simplices.shape[0],5))
    points[::2,0] = storm_data[(k_simplices[...,0]).astype(np.int32),0]
    points[::2,1] = storm_data[(k_simplices[...,0]).astype(np.int32),1]
    points[1::2,0] = storm_data[(k_simplices[...,1]).astype(np.int32),0]
    points[1::2,1] = storm_data[(k_simplices[...,1]).astype(np.int32),1]
    points[...,2] = np.repeat(k_simplices[...,2],2,)
    points[...,3] = np.repeat(k_simplices[...,3],2,)
    points[...,4] = np.repeat(k_simplices[...,4],2,)

    widget = gl.GLViewWidget()
    widget.show()
    item2 = rend.alpha_complex(filename = r"\Alpha")
    sizex = points[:,1].max()/32.2
    sizey = points[:,0].max()/32.2

    #widget.addItem(item2)

    item2.show()
    item2.set_data(position=points[...,0:2], simplices=points[...,2:5], alpha=np.float64(alpha_value ), size=float(1))
    item2.background_render(QPoint(sizey, sizex), 1.0)
    #item2.background_render(QPoint(1111,1361),1.0)
    #item2.background_render(QPoint(1059,1051),1.0) #render in fitting px size
    im = np.array(item2.image)
    #app.exec()
    return im

def create_storm(storm_data):
    widget = gl.GLViewWidget()
    widget.show()
    item2 = rend.points(filename=r"\STORM2")
    item2.show()
    item2.set_data(position=storm_data, size=float(20), maxEmission=256.0)
    item2.background_render(QPoint(1040, 1040), 1.0)
    im = np.array(item2.image)

    return im

def find_mapping(image, c_template, n_col=5, n_row=5, offset=0):
    results = []
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    template = cv2.cvtColor(c_template, cv2.COLOR_RGBA2GRAY)

    norm = np.linalg.norm(template)

    H = hough_transform()
    H.set_image(image)

    #im = im.astype("uint8")
    #imgs_t = []
    points1 = []
    points2 = []
    overlay = overlay.astype(np.uint16)
    for i in range(n_col):
        for j in range(n_row):
            k,l = j*200+offset,200+j*200+offset
            m,n = i*200+offset, 200+i*200+offset
            template_segment = template[k:l,m:n]
            #todo: fix out of bounds ... !!!!!!!!
            points1.append(np.array((k+template_segment.shape[0]/2,m+template_segment.shape[1]/2)))

            norm2 = np.linalg.norm(template_segment)
            if norm2< 0.15*norm:
                results.append([-1,np.array([-1,-1,-1,-1])])
                points2.append(np.array([0,0]))
                print("not enough data")
                continue
            imnew = c_template[k:l,m:n]
            H.set_template(template_segment)
            res = H.transform()
            results.append(res[0:2])
            points2.append(res[1][0:2])

            #imgs_t.append(imnew)
            #paint segments in image

            M = cv2.getRotationMatrix2D((imnew.shape[0] / 2, imnew.shape[1] / 2), res[0], 1)
            imnew = cv2.warpAffine(imnew, M, (imnew.shape[0], imnew.shape[1]))
            for h in range(template_segment.shape[0]):
                    for e in range(template_segment.shape[1]):
                        try:
                            overlay[int(2*res[1][0]-template_segment.shape[0]/2+h),int(2*res[1][1]-template_segment.shape[1]/2+e)] += imnew[h,e,0:4]
                        except Exception as error:
                            print('Caught this error: ' + repr(error))
    return points1,points2,overlay,results#todo:result wrapper


def error_management(result_list, points1, points2, n_row = 5):
    num = []
    for i,ent1 in enumerate(result_list):#todo: write function
        if ent1[1][-1] < 0:
            continue
        row1 = int(i/n_row)
        col1 = i%n_row
        max=0
        #print(i)
        for j,ent2 in enumerate(result_list):
            if ent2[1][-1]<0:
                continue
            row2 = int(j/n_row)
            col2 = j%n_row
            val1 = ent1[1][0]-(col1-col2)*100*np.cos(np.deg2rad(ent1[0]))+(row1-row2)*100*np.sin(np.deg2rad(ent1[0]))
            val2 = ent1[1][1]-(row1-row2)*100*np.cos(np.deg2rad(ent1[0]))-(col1-col2)*100*np.sin(np.deg2rad(ent1[0]))
            if np.absolute(val1-ent2[1][0])<25 and np.absolute(val2-ent2[1][1])<25:
                #print(True)
                max+=1
            #else:
                #print(False)
        if max>7:
            num.append(i)
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    points1 = points1[np.array(num)]
    points2 = points2[np.array(num)]
    p1 = np.fliplr(points1)
    p2 = np.float32(points2)*2
    p2 = np.fliplr(p2)
    print(p1, p2)

    return p1,p2

def test_pearson(sim, dstorm,mask, map):

    #mask = np.ones_like(dstorm).astype(np.float32)
    binary_mask = transform.warp(mask, inverse_map=map.inverse).astype(np.bool)


    #cv2.imshow("mask",mask.astype(np.uint8)*255)
    #cv2.imshow("mask_warp",binary_mask.astype(np.uint8)*255)
    #cv2.waitKey(0)
    dstorm_warped = transform.warp(dstorm, inverse_map=map.inverse)*255
    masked_dstorm = np.ma.array(data=dstorm_warped, mask=np.logical_not(binary_mask))

    masked_sim = np.ma.array(data=sim, mask=np.logical_not(binary_mask))
    #masked_mask = np.ma.array(data=binary_mask, mask=np.logical_not(binary_mask))
    corr_coef = pearsonr(masked_dstorm.flatten(),masked_sim.flatten())
    #cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\weighting\mask_dstorm.png", dstorm_warped.astype(np.uint8))

    #cv2.imshow("nasked",masked_dstorm)
    #cv2.imshow("masked", masked_sim)
    #corr_coef = np.ma.corrcoef(masked_dstorm.flatten(),masked_sim.flatten(), allow_masked=True)
    print("image:",corr_coef)#masked correlation only compare visible parts
    #cut_sim_masked = (sim*binary_mask).astype(np.uint8)
    #cv2.imwrite(r"C:\Users\biophys\Desktop\Masterarbeit\src\abb\weighting\mask_sim.png", cut_sim_masked.astype(np.uint8))
    #print("mask:",pearsonr(binary_mask.flatten()*255, cut_sim_masked.flatten()))#unmasked correlation to white
    return corr_coef