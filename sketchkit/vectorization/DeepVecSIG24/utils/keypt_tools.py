import numpy as np
import cv2
from skimage.feature import peak_local_max


def get_coords_from_heatmap(heatmap, dist = 2, pt_num = 50, thr_rel = 0.5, debug = False):
    '''
    Given,
        heatmap, 2D numpy array that could be a heatmap or a unsigned distance filed
            the rest parameters are not suggested to change, at least they work pretty well under current
            dectection task...
    Return,
        the keypoint extracted from the given input
    '''
    if debug:
        print("Log:\tgetting coords from UDF")

    # peak the local maximum from the heatmap, this is another candidate set of real keypoints
    coords = peak_local_max(heatmap, min_distance=dist, exclude_border = False, threshold_rel = thr_rel)
    if len(coords) > pt_num:
        coords = coords[np.argsort(heatmap[tuple(coords.T)])[-pt_num:]]
    if len(coords) > 0:
        coords[:, [0, 1]] = coords[:, [1, 0]]

    if debug:
        print("Log:\tDone")

    return coords


def to_img_coord(hm, dist = 2, pt_num = 50, thr_rel = 50):
    hm = hm.clip(0, 255)
    ### this output is xy ###
    cd = get_coords_from_heatmap(hm, dist, pt_num, thr_rel)
    return cd    

def udf_filter(udf, thr = 0.99):
    udf = udf.clip(0, 1)
    udf_mask = udf < thr
    res_mask = np.zeros(udf_mask.shape).astype(bool)
    _, regs = cv2.connectedComponents(udf_mask.astype(np.uint8), connectivity = 4)
    for r in np.unique(regs):
        if r == 0: continue
        m = regs == r
        if udf[m].min() > 0.1:
            res_mask[m] = True
    udf[res_mask] = 1
    return udf

def plot_udf_coord_numpy(sketch, udfs, udf_all_pt = None, alpha = 0.3, pt_num = 1000):
    '''
    Given,
        sketch, a numpy array as the input skech
        udfs, a list of numpy arrays as the input UDF(s)
    Return,
        a color image as numpy array that blends the sketch as colorized UDF and the detected keypoint from the UDF
    '''
    assert type(udfs) == list
    assert len(udfs) <= 6
    res = sketch.copy()
    # blend sketch with udf blend
    if udf_all_pt is not None:
        hm_all_pt = udf_to_hm(udf_all_pt)
        res = blend_heatmaps(res, hm_all_pt)
    # draw keypoint on the sketch
    res, keypts_dict = extract_keypts(udfs, pt_num, res)
    return res, keypts_dict

def udf_to_hm(udf):
    return 255 - udf.clip(0, 1).squeeze() * 255

def blend_heatmaps(img, hm, alpha = 0.3):
    # visualize the heatmap
    hm = cv2.applyColorMap(hm.astype(np.uint8), cv2.COLORMAP_HOT)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    # blend sketch and heatmap
    res = (img * (1 - alpha) + hm * alpha)
    return res.astype(np.uint8)

def extract_keypts(udfs, pt_numm, img = None, key_list = ["end_point", "sharp_turn", "junc"], 
        thr_list = [0.8, 0.8, 0.5]):
    assert len(udfs) <= len(key_list)
    # init colors
    green = np.array([0, 255, 0]) # end points, 0
    red = np.array([255, 0, 0]) # sharp turn, 1
    blue = np.array([0, 0, 255]) # T junctions, 2
    cyan = np.array([255, 255, 0]) # X junctions, 3
    yellow = np.array([0, 255, 255]) # star junctions, 4
    colors = [green, red, blue, cyan, yellow]
    # extract keypoint and overlay it on given image if possible
    keypts_dict = {}
    for i in range(len(udfs)):
        udf = udfs[i].clip(0, 1)
        # detect keypoint coordinates
        cd = to_img_coord((255 - udf * 255), pt_num = pt_numm, thr_rel = thr_list[i])
        # draw on the sketch
        if img is not None:
            img = draw_cross(img, cd, colors[i])
        keypts_dict[key_list[i]] = cd
    return img, keypts_dict

def draw_cross(res, cd, color):
    if len(cd) > 0:
        h, w = res.shape[0], res.shape[1]
        cd = cd.copy()
        # this function need hw format!
        cd[:, (0, 1)] = cd[:, (1, 0)]
        cd_center = tuple(cd.T)
        res[cd_center] = color
    return res
