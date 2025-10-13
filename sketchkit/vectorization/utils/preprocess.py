import numpy as np
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

