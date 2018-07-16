import numpy as np
from scipy.stats import pearsonr
from skimage import transform
import cv2

def test_pearson(sim,dstorm, map):
    sim = cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY)
    dstorm = cv2.cvtColor(dstorm, cv2.COLOR_RGB2GRAY)
    sim_auto_cut = cut_segments(sim, dstorm, (0,0), map)
    dstorm_warped = transform.warp(dstorm, inverse_map=map.inverse)
    cv2.imshow("sim", sim_auto_cut)
    cv2.imshow("storm", dstorm_warped)
    cv2.waitKey(0)
    pearson_auto = pearsonr(sim_auto_cut.flatten(), dstorm_warped.flatten())
    print(pearson_auto)
    #pearson_manuel = pearsonr(sim_auto_cut, dstorm_manuel)
    #todo: landmarks txt from cosidstorm map dstorm -> sim




def test_pearson(sim, dstorm, map):

    mask = np.ones_like(dstorm).astype(np.float32)
    binary_mask = transform.warp(mask, inverse_map=map.inverse).astype(np.bool)

    dstorm_warped = transform.warp(dstorm, inverse_map=map.inverse)
    masked_dstorm = np.ma.array(data=dstorm_warped, mask=np.logical_not(binary_mask))#

    masked_sim = np.ma.array(data=sim, mask=np.logical_not(binary_mask))
    masked_mask = np.ma.array(data=binary_mask, mask=np.logical_not(binary_mask))
    corr_coef = pearsonr(masked_dstorm.flatten(),masked_sim.flatten())
    print("image:",corr_coef)#masked correlation only compare visible parts
    cut_sim_masked = (sim*binary_mask).astype(np.uint8)
    #print("mask:",pearsonr(binary_mask.flatten()*255, cut_sim_masked.flatten()))#unmasked correlation to white
    return corr_coef

