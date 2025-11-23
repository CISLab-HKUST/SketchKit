from pathlib import Path as P
import sys
import os
directory = os.path.realpath(os.path.dirname(__file__))
directory = str(P(directory).parent)
if directory not in sys.path:
    sys.path.append(directory)

import torch
import time
import numpy as np
import cv2
import copy
from torch.nn import functional as F
from os import path

from svgpathtools import Path, Line, wsvg

from sketchkit.vectorization.DeepVecSIG24.utils.svg_tools import compute_dist_M
from scipy.signal import convolve2d
from scipy.ndimage import correlate
from skimage.morphology import skeletonize

def get_pt_mask(edge_map_x, edge_map_y):
    edge_map_x = edge_map_x.astype(bool)
    edge_map_y = edge_map_y.astype(bool)
    grid_mask_x = roll_edge(edge_map_x, -1, 'x').astype(int) + edge_map_x.astype(int)
    grid_mask_y = roll_edge(edge_map_y, -1, 'y').astype(int) + edge_map_y.astype(int)
    grid_mask_all = (grid_mask_x.astype(bool) | grid_mask_y.astype(bool)).astype(float)
    return grid_mask_x.astype(int), grid_mask_y.astype(int), grid_mask_all

def get_valence_np(edge_map_x, edge_map_y):
    edge_map_x = edge_map_x.astype(bool)
    edge_map_y = edge_map_y.astype(bool)
    valence_x, valence_y, pt_mask = get_pt_mask(edge_map_x, edge_map_y)

    # sum up all valences for each grid
    valence_map = valence_x + valence_y
    assert valence_map.max() <= 4

    # should we adjust the counter numbers accrodingly?
    return valence_map, pt_mask

def init_base_coord(pt_map, device, gsizes):
    '''
    Given:
        pt_map, batch x 2 x height x width tensor, stores the center point of each UDF grid
        device, string which should be "cuda" or "cpu", decide where to store this base coord map
    Action:
        return base_coord
    Note:
        height and width from the corresponding UDF's shape, remember that 4 UDF points forms on grid
        therefore the point map size should always be height -1 and width -1
    '''
    b, _, h, w = pt_map.shape
    base_coord = []
    for gsize in gsizes:
        gsize = float(gsize)
        hg = int(h * gsize)
        wg = int(w * gsize)
        assert hg == h * gsize
        assert wg == w * gsize
        xs = torch.linspace(start = gsize / 2, end = wg - gsize / 2, steps = w) 
        ys = torch.linspace(start = gsize / 2, end = hg - gsize / 2, steps = h)
        xx, yy = torch.meshgrid(xs, ys, indexing = 'xy')
        base_coord.append(torch.stack((xx, yy), dim = -1).unsqueeze(0).permute(0, 3, 1, 2))
    base_coord = torch.cat(base_coord, dim = 0)
    return base_coord.to(device)

def pre_to_map(edge_pre, stroke_mask = None):
    '''
    Given:
        edge_pre:   A int tensor with shape (b, h, w, 4) or (b, h, w, 1), the last dimension stores the edge flags, 
                    this function support both the edge values from network prediction tensor or the ground truth tensor 
                    , but only the first tensor format will not block the gradient

                    the value defination of edge flag ground truth:
                    0: None, 1: edge flag along x-axis, 2: edge flag along y-axis, 3: edge flag along both axis
                    
                    PLEASE NOTE: the edge flag along x-axis will produce a polyline in y-axis direction! cause the polyline will
                    go across the grid edge. this could be kind of confusing...
    Return:
        edge_map_x: A float tensor with shape (b, h, w, 1), the edge flag along x-axis
        edge_map_y: A float tensor with shape (b, h, w, 1), the edge flag along y-axis
        edge_map:   A float tensor with shape (b, h, w, 4), the full edge flag in one hot vector format
    '''
    softmax = torch.nn.Softmax(dim = -1)
    relu = torch.nn.ReLU()

    # check which type of the edge flag is
    c = edge_pre.shape[-1]
    if c == 4:
        edge_map_temp = softmax(edge_pre)
        edge_mask = edge_map_temp.argmax(dim = -1)
        edge_mask = F.one_hot(edge_mask, num_classes=4)
        
        # https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836
        # push the choosen value (the edge flag prediction) greater than 1 and the rest less than 0
        edge_map_temp = (edge_map_temp + 2) * edge_mask +  edge_map_temp * (1 - edge_mask)
        edge_map_temp = edge_map_temp - 1
        
        # we have to do this to avoid inplace tensor assignment
        edge_map = torch.clamp(relu(edge_map_temp), min = 0, max = 1)
        edge_map_x = edge_map[...,1] + edge_map[...,3]
        edge_map_y = edge_map[...,2] + edge_map[...,3]

    
    elif c == 2:
        edge_mask = (torch.sigmoid(edge_pre) > 0.5).float()
        edge_prediction = torch.sigmoid(edge_pre)
        edge_prediction = (edge_prediction + 1) * edge_mask + (edge_prediction - 1) * (1 - edge_mask)
        edge_map = torch.clamp(relu(edge_prediction), min = 0, max = 1)
        edge_map_x = edge_map[...,0]
        edge_map_y = edge_map[...,1]
    
    else:
        # generate edges from GT, this branch will break the gradient
        assert c == 1
        edge_map = edge_pre.squeeze(dim = -1).long()
        edge_map_x = torch.logical_or((edge_map == 1), (edge_map == 3)).float()
        edge_map_y = torch.logical_or((edge_map == 2), (edge_map == 3)).float()
        edge_map = F.one_hot(edge_map, num_classes = 4).float()
    
    if stroke_mask is not None:
        edge_map_x = edge_map_x * stroke_mask.squeeze().long()
        edge_map_y = edge_map_y * stroke_mask.squeeze().long()
    return edge_map_x, edge_map_y, edge_map

def align_keypt(pt_map, key_pts, valence_map, val, pt_type, idx_to_keypts, thr = 1.5):
    ndc_pts = pt_map[np.where(valence_map == val)].reshape(-1, 2)
    if len(ndc_pts) == 0:
        return key_pts
    pt_idxs = np.array(np.where(valence_map == val)).T # idx are in (i, j) format
    
    # compute distance matrix between to point groups
    # row is the key points
    # column is the ndc grid points
    M_dist = compute_dist_M(ndc_pts, key_pts)
    
    # for each DC vertex, find the closest keypoint
    # M_idx = np.argsort(M_dist, axis = 0)
    M_idx = np.argmin(M_dist, axis = 0)
    # assert ( M_idx[0,:] == np.argmin(M_dist, axis = 0) ).all()

    # get the closest distance for each keypoint
    ndc_pts_update = []
    match_idx = M_dist[M_idx, np.arange(M_dist.shape[1])]
    for i in range(len(match_idx)):
        # update point type map if we find one match
        if match_idx[i] < thr:
            ndc_idx = M_idx[i]
            pt_type[tuple(pt_idxs[ndc_idx].T)] = val
            # this key point is save as (x, y) format
            idx_to_keypts[tuple(pt_idxs[ndc_idx].T)] = key_pts[i]
            ndc_pts_update.append(pt_map[tuple(pt_idxs[ndc_idx].T)])
    return np.array(ndc_pts_update)

def gen_sample_idx(size, ratio, head = True, offset = 0):
    # location could be: head/tail
    assert size % ratio == 0
    if head:
        return np.arange(0 + offset, size, ratio)
    else:
        return np.arange(ratio - (1 + offset), size, ratio)
    
def roll_by_axis(array, shift = -1, axis = 'x'):
    assert len(array.shape) == 2
    assert shift < 0
    arr = array.copy()
    if axis == 'x':
        arr[0:abs(shift), :] = False
        return np.roll(arr, shift, axis = 0)
    elif axis == 'y':
        arr[:, 0:abs(shift)] = False
        return np.roll(arr, shift, axis = 1)
    else:
        raise ValueError("Axis should only be 'x' or 'y' while get %s"%axis)

def apply_kernel_biase(kernel, axis, size):
    if axis == 'x':
        mask = np.zeros(kernel.shape).astype(bool)
        mask[:, 0:int(size/2)] = True
        kernel[mask] = kernel[mask] * 2
    elif axis == 'y':
        mask = np.zeros(kernel.shape).astype(bool)
        mask[0:int(size/2), :] = True
        kernel[mask] = kernel[mask] * 2
    else:
        assert axis is None
    return kernel

# https://stackoverflow.com/questions/56948729/how-to-create-a-triangle-kernel-in-python
def triangle_kernel(size = 9, biased = None):
    r = np.arange(size)
    kernel1d = (size + 1 - np.abs(r - r[::-1])) / 2
    kernel2d = np.outer(kernel1d, kernel1d)
    kernel2d = apply_kernel_biase(kernel2d, biased, size)
    kernel2d = kernel2d / kernel2d.sum()
    return kernel2d

def downsample_by_axis(in_array, ratio = 2, axis = 'all', return_sum = True, head = True):
    h, w = in_array.shape[0], in_array.shape[1]
    assert h % ratio == 0
    assert w % ratio == 0
    h_new = int(h / ratio)
    w_new = int(w / ratio)
    if axis == 'all':
        down_all = in_array.reshape(h, w_new, ratio).transpose(0,2,1).reshape(h_new, ratio ** 2, w_new).transpose(0,2,1)
        return down_all.sum(axis = -1) if return_sum else down_all
    elif axis == 'x':
        # np.arange(0, h, ratio)
        down_x = in_array[gen_sample_idx(h, ratio, head), :].reshape(h_new, w_new, ratio)
        return down_x.sum(axis = 2) if return_sum else down_x
    elif axis == 'y':
        # np.arange(0, w, ratio)
        down_y = in_array[:, gen_sample_idx(w, ratio, head)].reshape(h_new, ratio, w_new)
        return down_y.sum(axis = 1) if return_sum else down_y
    else:
        raise ValueError("axis could only be one of ['all', 'x', 'y'] but not %s"%axis)

def expand_xy(array, mode = 'xy'):
    if mode == 'x':
        array_ = roll_by_axis(array, axis = 'x')
    elif mode == 'y':
        array_ = roll_by_axis(array, axis = 'y')
    elif mode == 'xy':
        array_ = roll_by_axis(array, axis = 'x') | roll_by_axis(array, axis = 'y')
    else:
        raise ValueError("expand mode only support one of the following options: 'x', 'y', 'xy' while got mode %s", mode)
    return array | array_

def get_downsampled_idx(h, w, ratio):
    iis = np.arange(h)
    jjs = np.arange(w)
    iiv, jjv = np.meshgrid(iis, jjs, indexing = 'ij')
    iiv = downsample_by_axis(iiv, ratio, return_sum = False)
    jjv = downsample_by_axis(jjv, ratio, return_sum = False)
    return iiv.flatten(), jjv.flatten()

def downsample_all(edge_map_x, edge_map_y, pt_map, usm, valence_map, pt_type, lines_map_x = None, 
    lines_map_y = None, ratio = 2, return_lines = True):
    kernel = triangle_kernel(size = 3)
    assert edge_map_x.shape == edge_map_y.shape
    assert edge_map_x.shape == usm.shape
    assert edge_map_x.shape == (pt_map.shape[0], pt_map.shape[1])
    h, w = edge_map_x.shape
    assert h % ratio == 0
    assert w % ratio == 0
    h_down = int(h / ratio)
    w_down = int(w / ratio)
    # expand usm to algin to its downsized version
    usm_pre_down = downsample_by_axis(usm, ratio, 'all') # USM from prediction
    # Image.fromarray(usm.astype(bool)).save("usm_pre_2x.png")
    usm_pre_down = usm_pre_down > 0 # we DON'T need to extend USM maps from the prediction, it has already done so.
    # Image.fromarray(usm_pre_down).save("usm_pre.png")
    # sample edge map along x-axis
    x_down = downsample_by_axis(edge_map_x, ratio, 'x')
    edge_x_down = x_down >= 1
    # edge_x_down = downsample_edge(edge_map_x, edge_map_y, 'x', ratio)
    # edge_x_down = downsample_by_shift(edge_map_x, axis = 'x', ratio = ratio)
    # edge_x_down = downsample_by_conv(edge_map_x, kernel, axis = 'x', ratio = ratio)
    # sample edge map along y-axis
    y_down = downsample_by_axis(edge_map_y, ratio, 'y')
    edge_y_down = y_down >=1
    # edge_y_down = downsample_edge(edge_map_x, edge_map_y, 'y', ratio)
    # edge_y_down = downsample_by_shift(edge_map_y, axis = 'y', ratio = ratio)
    # edge_y_down = downsample_by_conv(edge_map_y, kernel, axis = 'y', ratio = ratio)
    ## compute all USM brought by the downsampling
    valence_down = downsample_by_axis(valence_map, ratio, 'all', return_sum = False)
    pt_type_down = downsample_by_axis(pt_type, ratio, 'all', return_sum = False)
    top_down = x_down
    bottom_down = roll_by_axis(x_down, axis = 'x')
    left_down = y_down
    right_down = roll_by_axis(y_down, axis = 'y')
    # consider any grid contains more than 1 endpoints, or 1 endpoint with any other keypoints as under sampled region 
    usm_valence1 = ((pt_type_down == 1).sum(axis = -1) >= 1)
    usm_valence2 = (pt_type_down == 2).all(axis = -1)
    usm_left_down = (left_down > 1) | usm_valence1 & ~usm_valence2
    usm_top_down = (top_down > 1) | usm_valence1 & ~usm_valence2
    usm_bottom_down = (bottom_down > 1) | usm_valence1 & ~usm_valence2
    usm_right_down = (right_down > 1) | usm_valence1 & ~usm_valence2
    # usm_left_down = left_down > 1
    # usm_top_down = top_down > 1
    # usm_bottom_down = bottom_down > 1
    # usm_right_down = right_down > 1
    usm_4way_down = ((top_down >= 1).astype(int) + (bottom_down >= 1).astype(int) + 
        (left_down >= 1).astype(int) + (right_down >= 1).astype(int)) == 4
    usm_xy_down = (top_down + left_down) > 1
    # make sure the boundary of USM won't contain any under sampling cases
    usm_top_down = expand_xy(usm_top_down, 'x')
    usm_left_down = expand_xy(usm_left_down, 'y')
    usm_4way_down = expand_xy(usm_4way_down, 'xy')
    usm_downsample = usm_left_down | usm_right_down | usm_top_down | usm_bottom_down | usm_4way_down | usm_xy_down
    # check every USM region if it is really contains important structure (contain keypoints)
    usm_ = cv2.resize(usm_downsample.astype(np.uint8), (w, h), interpolation = cv2.INTER_NEAREST).astype(bool)
    _, regions = cv2.connectedComponents(usm_.astype(np.uint8), connectivity = 4)
    for r in np.unique(regions):
        if r == 0: continue
        r_mask = regions == r
        # drop current USM region if it doesn't contain any keypoint or it is too large
        if (pt_type[r_mask] > 0).sum() == 0:
          r_mask = cv2.resize(r_mask.astype(np.uint8), (w_down, h_down), interpolation = cv2.INTER_NEAREST)
          usm_downsample[r_mask] = False
    # Image.fromarray(usm_downsample).save("usm_down.png")
    usm_down = usm_pre_down | usm_downsample
    # usm_down = usm_pre_down
    ## sample pt map
    '''
    this will be kind of difficult...
    so we will need to have a new key point map and a new dual contouring reconstruction method
    '''
    # generate the resmaple idx
    # setting the edge flag by USM
    if return_lines:
        edge_x_down[usm_down] = False
        edge_y_down[usm_down] = False
        # the logic here is really annoying!
        # get line mask
        line_mask = cv2.resize(usm_down.astype(np.uint8), (w, h), interpolation = cv2.INTER_NEAREST).astype(bool)
        usm_down_x = (expand_xy(usm_down, 'x') | usm_down).copy()
        usm_down_y = (expand_xy(usm_down, 'y') | usm_down).copy()
        line_mask_x = cv2.resize(usm_down_x.astype(np.uint8), (w, h), interpolation = cv2.INTER_NEAREST).astype(bool)
        line_mask_y = cv2.resize(usm_down_y.astype(np.uint8), (w, h), interpolation = cv2.INTER_NEAREST).astype(bool)
        line_mask_x_base = line_mask_x.copy()
        line_mask_y_base = line_mask_y.copy()
        line_mask_x[gen_sample_idx(h, ratio, True), :] = False
        line_mask_y[:, gen_sample_idx(w, ratio, True)] = False
        line_mask = line_mask_x | line_mask_y | line_mask
        line_mask_x = line_mask | line_mask_y_base
        line_mask_y = line_mask | line_mask_x_base
        # Image.fromarray(line_mask_x).save("line_mask_x.png")
        # Image.fromarray(line_mask_y).save("line_mask_y.png")
        # Image.fromarray(line_mask).save("line_mask.png")
    ## compute downsampled center point map
    # generate center point map of each super grid
    # edge_x_down_ = edge_x_down.copy()
    # edge_x_down_[0,...] = False
    # edge_y_down_ = edge_y_down.copy()
    # edge_y_down_[..., 0] = False
    # pt_mask_center_ = edge_x_down|np.roll(edge_x_down_, shift = -1, axis = 0)|edge_y_down|np.roll(edge_y_down_, shift=-1, axis=1)
    pt_mask_center_2x = valence_map > 0
    pt_mask_center_2x_v1 = valence_map == 1
    pt_mask_center_2x_v3 = valence_map > 2
    pt_count_center = downsample_by_axis(pt_mask_center_2x, ratio)
    pt_mask_center = pt_count_center > 0 

    pt_count_center_v1 = downsample_by_axis(pt_mask_center_2x_v1, ratio)
    pt_mask_center_v1 = pt_count_center_v1 > 0
    
    pt_count_center_v1[pt_count_center_v1 == 0] = 1
    pt_map_masked_2x = pt_map * (pt_mask_center_2x)[..., np.newaxis]
    pt_map_masked_2x_v1 = pt_map * (pt_mask_center_2x_v1)[..., np.newaxis]
    iiv, jjv = get_downsampled_idx(h, w, ratio)
    
    pt_mask_center_v3 = downsample_by_axis(pt_mask_center_2x_v3, ratio) > 0
    pt_mask_center_v1 = np.logical_xor(pt_mask_center_v1, pt_mask_center_v3) & pt_mask_center_v1
    pt_map_down_center_v1 = pt_map_masked_2x_v1[(iiv, jjv)].reshape(-1, ratio**2, 2).sum(axis = 1).reshape(h_down, w_down, -1) / pt_count_center_v1[...,np.newaxis]
    pt_count_center[~pt_mask_center] = 1
    pt_map_down_center = pt_map_masked_2x[(iiv, jjv)].reshape(-1, ratio**2, 2).sum(axis = 1).reshape(h_down, w_down, -1) / pt_count_center[...,np.newaxis]
    pt_map_down_center = pt_map_down_center * (~pt_mask_center_v1)[..., np.newaxis] + pt_map_down_center_v1 * pt_mask_center_v1[..., np.newaxis]
    
    # pt_mask_center_add = (np.logical_xor(pt_mask_center_, pt_mask_center.astype(bool)) & pt_mask_center_)|(np.logical_xor(pt_mask_center_, pt_mask_center.astype(bool)) & pt_mask_center)
    # pt_map_down_center_add = pt_map[(iiv, jjv)].reshape(-1, ratio**2, 2).mean(axis = 1).reshape(h_down, w_down, -1) * pt_mask_center_add[...,np.newaxis]
    # pt_map_down_center = pt_map_down_center + pt_map_down_center_add
    if return_lines:
        # update top point maps
        pt_mask_top_2x = edge_map_x
        pt_map_top = pt_map * (pt_mask_top_2x)[..., np.newaxis]
        pt_mask_top = downsample_by_axis(pt_mask_top_2x, ratio, axis = 'x', head = True)
        pt_mask_top[pt_mask_top == 0] = 1
        pt_map_down_top = pt_map_top[gen_sample_idx(h, ratio, True), :, ...].reshape(h_down, w_down, ratio, 2).sum(axis = 2) / pt_mask_top[...,np.newaxis]
        # update left point maps
        pt_mask_left_2x = edge_map_y
        pt_map_left = pt_map * (pt_mask_left_2x)[..., np.newaxis]
        pt_mask_left = downsample_by_axis(pt_mask_left_2x, ratio, axis = 'y', head = True)
        pt_mask_left[pt_mask_left == 0] = 1
        pt_map_down_left = pt_map_left[:, gen_sample_idx(w, ratio, True), ...].reshape(h_down, ratio,w_down, 2).transpose(0,2,1,3).sum(axis = 2) / pt_mask_left[...,np.newaxis]
        # update bottom point maps
        pt_mask_bottom_2x = roll_by_axis(edge_map_x, axis = 'x')
        pt_map_bottom = pt_map * (pt_mask_bottom_2x)[..., np.newaxis]
        pt_mask_bottom = downsample_by_axis(pt_mask_bottom_2x, ratio, axis = 'x', head = False)
        pt_mask_bottom[pt_mask_bottom == 0] = 1
        pt_map_down_bottom = pt_map_bottom[gen_sample_idx(h, ratio, False), :, ...].reshape(h_down, w_down, ratio, 2).sum(axis = 2) / pt_mask_bottom[...,np.newaxis]    
        # update right point maps
        pt_mask_right_2x = roll_by_axis(edge_map_y, axis = 'y')
        pt_map_right = pt_map * (pt_mask_right_2x)[..., np.newaxis]
        pt_mask_right = downsample_by_axis(pt_mask_right_2x, ratio, axis = 'y', head = False)
        pt_mask_right[pt_mask_right == 0] = 1
        pt_map_down_right = pt_map_right[:, gen_sample_idx(w, ratio, False), ...].reshape(h_down, ratio, w_down, 2).transpose(0,2,1,3).sum(axis = 2) / pt_mask_right[...,np.newaxis]
        # extract the polylines need to be reserved
        
        l1 = lines_map_x[line_mask_x]
        l1 = np.concatenate(l1[l1 != False])
        l2 = lines_map_y[line_mask_y]
        l2 = np.concatenate(l2[l2 != False])
        lines_reserved = np.concatenate((l1, l2), axis = 0)
        
        # lines_to_svg(lines_reserved, 512, 512, "line_reserved.svg")
        # reconstruct from the downsampled UDF
        edge_map_down = cut_branch(np.stack((edge_x_down, edge_y_down), axis = -1))
        edge_x_down = edge_map_down[..., 0]
        edge_y_down = edge_map_down[..., 1]
        usm_down = expand_xy(usm_down)
        lines_down = []
        for i in range(h_down):
            for j in range(w_down):
                # test x-axis direction
                if edge_x_down[i][j] and i != 0:
                    # at the top USM boundary
                    if usm_down[i - 1][j] == False and usm_down[i][j] and (pt_map_down_center[i - 1][j] != pt_map_down_top[i][j]).any():
                        lines_down.append(pt_map_down_center[i - 1][j])
                        lines_down.append(pt_map_down_top[i][j])
                    # at the bottom USM boundary
                    elif usm_down[i - 1][j] and usm_down[i][j] == False and (pt_map_down_bottom[i - 1][j] != pt_map_down_center[i][j]).any():
                        lines_down.append(pt_map_down_bottom[i - 1][j])
                        lines_down.append(pt_map_down_center[i][j])
                    # no USM boudnary
                    elif usm_down[i - 1][j] == False and usm_down[i][j] == False and (pt_map_down_center[i - 1][j] != pt_map_down_center[i][j]).any():
                        lines_down.append(pt_map_down_center[i - 1][j])
                        lines_down.append(pt_map_down_center[i][j])
                    elif usm_down[i - 1][j] and usm_down[i][j] and (pt_map_down_bottom[i - 1][j] != pt_map_down_top[i][j]).any():
                        lines_down.append(pt_map_down_bottom[i - 1][j])
                        lines_down.append(pt_map_down_top[i][j])    
                if edge_y_down[i][j] and j != 0: # edge map y
                    # at the left USM boundary
                    if usm_down[i][j - 1] == False and usm_down[i][j] and (pt_map_down_center[i][j - 1] != pt_map_down_left[i][j]).any():
                        lines_down.append(pt_map_down_center[i][j - 1])
                        lines_down.append(pt_map_down_left[i][j])
                    # at the right USM boundary
                    elif usm_down[i][j - 1] and usm_down[i][j] == False and (pt_map_down_right[i][j - 1] != pt_map_down_center[i][j]).any():
                        lines_down.append(pt_map_down_right[i][j - 1])
                        lines_down.append(pt_map_down_center[i][j])
                    # no USM boudnary
                    elif usm_down[i][j - 1] == False and usm_down[i][j] == False and (pt_map_down_center[i][j - 1] != pt_map_down_center[i][j]).any():
                        lines_down.append(pt_map_down_center[i][j - 1])
                        lines_down.append(pt_map_down_center[i][j])
                    elif usm_down[i][j - 1] and usm_down[i][j] and (pt_map_down_right[i][j - 1] != pt_map_down_left[i][j]).any():
                        lines_down.append(pt_map_down_right[i][j - 1])
                        lines_down.append(pt_map_down_left[i][j])
        # combine all results and return
        lines_down = np.array(lines_down)
        # lines_to_svg(lines_down, 512, 512, "line_down.svg")
        # lines_res = lines_down
        lines_res = np.concatenate((lines_reserved, lines_down), axis = 0)
        line = lines_down

        return lines_res
    else:
        return edge_x_down, edge_y_down, usm_down, pt_map_down_center

def refine_topology(edge_map, pt_map, under_sampled_map, lines_map_x, 
    lines_map_y, keypts, 
    tensor_input = False, 
    down_rate = 2, 
    downsample = True, 
    manual = False,
    full_auto_mode = False):
    '''
    lines_map_x, line_map_y will be inplace modified
    '''
    kernel = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]], dtype = np.uint8)
    if tensor_input:
        assert manual == False
        edge_map = edge_map.squeeze().to(bool).detach().cpu().numpy()
        pt_map = pt_map.squeeze().detach().cpu().numpy()
        under_sampled_map = under_sampled_map.squeeze().to(bool).detach().cpu().numpy()
    h, w = edge_map.shape[0], edge_map.shape[1]
    long_side = h if h > w else w
    ## register keypoints to pt map
    edge_map_x = edge_map[..., 0]
    edge_map_y = edge_map[..., 1]

    valence_map, pt_mask = get_valence_np(edge_map_x, edge_map_y)
    pt_type = np.zeros(valence_map.shape)
    idx_to_keypts = {}

    # align key points and update them back to dict
    if manual == False:
        keypts['end_point'] = align_keypt(pt_map, keypts['end_point'], valence_map, 1, pt_type, idx_to_keypts, 3.5)
        keypts['sharp_turn'] = align_keypt(pt_map, keypts['sharp_turn'], valence_map, 2, pt_type, idx_to_keypts, 3.5)
        # some times there could exists T or X junction but with just valence 2 vertexs
        align_keypt(pt_map, keypts['junc'], valence_map, 2, pt_type, idx_to_keypts)
        keypts['junc'] = align_keypt(pt_map, keypts['junc'], valence_map, 3, pt_type, idx_to_keypts, 3.5)
    # for the predicted keypoints, they should have been aligned, 
    # for the new keypoint, they are user input, should be accurate
    # therefore, there is no need to do the alignment
    # else:
    #     align_keypt(pt_map, keypts['end_point'], valence_map, 1, pt_type, idx_to_keypts, 3.5)
    #     align_keypt(pt_map, keypts['sharp_turn'], valence_map, 2, pt_type, idx_to_keypts, 3.5)
    #     align_keypt(pt_map, keypts['junc'], valence_map, 2, pt_type, idx_to_keypts)
    #     align_keypt(pt_map, keypts['junc'], valence_map, 3, pt_type, idx_to_keypts, 3.5)

    # remove possible holes
    if manual == False:
        _, regions_raw = cv2.connectedComponents(under_sampled_map.astype(np.uint8), connectivity = 4)
        under_sampled_map = cv2.dilate( under_sampled_map.astype(np.uint8), kernel, iterations = 1 )
        under_sampled_map = cv2.erode( under_sampled_map.astype(np.uint8), kernel, iterations = 1 )
        # split under sampled map into individual regions
        _, regions = cv2.connectedComponents(under_sampled_map.astype(np.uint8), connectivity = 4)
    else:
        regions = under_sampled_map

    under_sampled_map_applied = np.zeros(under_sampled_map.shape).astype(int)
    under_sampled_map_uncertain = np.zeros(under_sampled_map.shape).astype(int)

    # for each region reconstruct the topology
    if len(np.unique(regions)) == 1: 
        lines = linemap_to_lines(lines_map_x, lines_map_y)
        return lines, {}, under_sampled_map, under_sampled_map_uncertain
    
    # record current region need usm surgery
    refined_lines = []
    refined_dict = {}
    for idx in np.unique(regions):
        if idx == 0: continue # skip background
        # get the key point (if have) from the given region
        region = regions == idx
        # skip if the USM is too large or it connect two USM regions
        if manual == False:
            r = np.unique(regions_raw[region])
            r = r[r!=0]
            if (region.sum() > long_side / 50) or len(r) > 1:
                continue
        under_sampled_map_updated = np.zeros(under_sampled_map.shape, dtype = bool)
        under_sampled_map_updated[region] = True
        # get the connection points from the given region
        connection_pts = find_connection_points(valence_map != 0, pt_map, region.astype(int) * 255, under_sampled_map.astype(bool))
        if connection_pts == False: continue
        # update the region topology
        applied, added_lines = update_line_maps(lines_map_x, lines_map_y, pt_map, pt_mask, pt_type, connection_pts, region, 
            refined_lines, idx_to_keypts, under_sampled_map_updated, manual, keypts, full_auto_mode)
        if applied is not None:
            if applied:
                under_sampled_map_applied[under_sampled_map_updated] = idx
                refined_dict[idx] = added_lines
            else:
                under_sampled_map_uncertain[under_sampled_map_updated] = idx

    # it is unsafe to downsample SVG under manual mode, cause now the USM is probably not complete
    if downsample:
        start = time.time()
        lines = downsample_all(edge_map_x, edge_map_y, pt_map, regions != 0, 
            valence_map, pt_type, lines_map_x, lines_map_y, ratio = down_rate)
        end = time.time()
        # print("log:\tdual contouring downsampling finished in %.2f seconds"%(end-start))
    else:
        lines = linemap_to_lines(lines_map_x, lines_map_y)    

    if len(refined_lines) > 0:
        assert len(refined_lines) % 2 == 0
        for i in range(len(refined_lines)-1, -1, -2):
            if len(refined_lines[i].shape) < 1 or len(refined_lines[i - 1].shape) < 1:
                refined_lines.pop(i)
                refined_lines.pop(i-1)
        refined_lines = np.stack(refined_lines, axis = 0)
        if len(refined_lines) > 0:
            lines = np.concatenate((lines, refined_lines), axis = 0)
    return lines, refined_dict, under_sampled_map_applied, under_sampled_map_uncertain

def pt_num(pt):
    if len(pt.shape) == 1:
        return 0 if pt.shape[0] == 0 else 1
    if len(pt.shape) == 2: 
        return len(pt)

def connect_usm_pts(connection_pts, pt, added_lines):
    assert pt_num(pt) == 1
    for i in range(len(connection_pts)):
        added_lines.append(connection_pts[i][0])
        added_lines.append(pt)

def update_line_maps(lines_map_x, lines_map_y, pt_map, pt_mask, pt_type, connection_pts, 
        region, refined_lines, idx_to_keypts, 
        usm_updated = None, 
        manual = False, 
        keypts = None,
        full_auto_mode = False):
    # this is a inplace function!
    # rebuild the topology inside this region
    # get the region valence
    valence = len(connection_pts)
    added_lines = []
    if manual and full_auto_mode == False:
        assert keypts is not None
        end, sharp, junc, avg_pt = query_keypt_by_region_no_alginment(keypts, region.astype(bool), pt_map, pt_mask)
    else:
        end, sharp, junc, avg_pt = query_keypt_by_region(pt_type, region.astype(bool), idx_to_keypts, pt_map, pt_mask)
    manual = manual or full_auto_mode
    need_update = False
    if valence == 0: 
        return None, None
    # end point
    elif valence == 1:
        if pt_num(end) == 1 and (pt_num(junc) == 0 or manual):
            connect_usm_pts(connection_pts, end, added_lines)
            need_update = True
    elif valence == 2:
        need_update = True
        if pt_num(sharp) == 1 and (pt_num(end) == 0 or manual):
            connect_usm_pts(connection_pts, sharp, added_lines)
        elif pt_num(junc) == 1 and (pt_num(end) == 0 or manual):
            connect_usm_pts(connection_pts, junc, added_lines)
        elif pt_num(end) >= 1 and pt_num(sharp) >= 1 and pt_num(junc) >=1:
            s = sharp if pt_num(sharp) == 1 else sharp[0]
            connect_usm_pts(connection_pts, s, added_lines)
        elif pt_num(sharp) > 1:
            need_update = False
        elif pt_num(end) > 0 and manual == False:
            need_update = False
        elif manual:
            added_lines.append(connection_pts[0][0])
            added_lines.append(connection_pts[1][0])
        else:
            need_update = False

    # T, Y junction
    elif valence == 3:
        need_update = True
        if pt_num(junc) == 1 and (pt_num(end) == 0 or manual):
            connect_usm_pts(connection_pts, junc, added_lines)
        elif pt_num(sharp) == 1 and (pt_num(end) == 0 or manual):
            connect_usm_pts(connection_pts, sharp, added_lines)
        elif pt_num(end) >= 1 and pt_num(sharp) >= 1 and pt_num(junc) >=1:
            j = junc if pt_num(junc) == 1 else junc[0]
            connect_usm_pts(connection_pts, j, added_lines)
        elif pt_num(junc) > 1:
            need_update = False
        elif len(end) > 0 and manual == False:
            need_update = False
        elif manual:
            # for i in range(len(connection_pts)):
            #     added_lines.append(connection_pts[i][0])
            #     added_lines.append(avg_pt)
            dir_matrix = get_direction_matrix(connection_pts)
            assert (3, 3) == dir_matrix.shape
            pt_idx01 = np.unravel_index(np.argmax(dir_matrix), dir_matrix.shape)
            assert len(pt_idx01) == 2
            pt_idx2 = [i for i in [0, 1, 2] if i not in pt_idx01]
            pt0 = connection_pts[pt_idx01[0]][0]
            pt1 = connection_pts[pt_idx01[1]][0]
            pt2 = connection_pts[pt_idx2[0]][0]
            pt3 = (pt0 + pt1) / 2
            added_lines.append(pt0)
            added_lines.append(pt1)
            added_lines.append(pt2)
            added_lines.append(pt3)
        else:
            need_update = False

    # X junction
    elif valence == 4:
        need_update = True
        if pt_num(junc) == 1 and (len(end) == 0 or manual):
            connect_usm_pts(connection_pts, junc, added_lines)
        elif pt_num(sharp) == 1 and (pt_num(end) == 0 or manual):
            connect_usm_pts(connection_pts, sharp, added_lines)
        elif pt_num(end) >= 1 and pt_num(sharp) >= 1 and pt_num(junc) >=1:
            j = junc if pt_num(junc) == 1 else junc[0]
            connect_usm_pts(connection_pts, j, added_lines)
        elif (pt_num(junc) > 1 or len(end) > 0) and manual == False:
            need_update = False
        elif manual:
            try:
                tans_edge = []
                # let's emumerate all possible connections here
                # (0, 1), (2, 3)
                line0 = Line(complex(*connection_pts[0][0]), complex(*connection_pts[1][0]))
                line1 = Line(complex(*connection_pts[2][0]), complex(*connection_pts[3][0]))
                tans_edge.append(line0.unit_tangent(0))
                tans_edge.append(line1.unit_tangent(0))
                # (1, 2), (0, 3)
                line0 = Line(complex(*connection_pts[1][0]), complex(*connection_pts[2][0]))
                line1 = Line(complex(*connection_pts[0][0]), complex(*connection_pts[3][0]))
                tans_edge.append(line0.unit_tangent(0))
                tans_edge.append(line1.unit_tangent(0))
                # (0, 2), (1, 3)
                line0 = Line(complex(*connection_pts[0][0]), complex(*connection_pts[2][0]))
                line1 = Line(complex(*connection_pts[1][0]), complex(*connection_pts[3][0]))
                tans_edge.append(line0.unit_tangent(0))
                tans_edge.append(line1.unit_tangent(0))
                tans_stroke = []
                for i in range(4):
                    tans_stroke.append(connection_pts[i][1])
                dir_matrix = np.zeros((6, 4))
                for i in range(6):
                    for j in range(4):
                        dir_matrix[i][j] = complex_dot(tans_edge[i], tans_stroke[j])
                dir_matrix = dir_matrix.mean(axis = 1).squeeze()
                parallel_idx = np.argsort(dir_matrix)
                group_list = [((0, 1), (2, 3)), ((1, 2), (0, 3)), ((0, 2), (1, 3))]

                if parallel_idx[-1] == 0 and dir_matrix[1] > 0.9 or parallel_idx[-1] == 1 and dir_matrix[0] > 0.9:
                    group_a = group_list[0][0]
                    group_b = group_list[0][1]

                elif parallel_idx[-1] == 2 and dir_matrix[3] > 0.9 or parallel_idx[-1] == 3 and dir_matrix[2] > 0.9:
                    group_a = group_list[1][0]
                    group_b = group_list[1][1]

                elif parallel_idx[-1] == 4 and dir_matrix[5] > 0.9 or parallel_idx[-1] == 5 and dir_matrix[4] > 0.9:
                    group_a = group_list[2][0]
                    group_b = group_list[2][1]
                update_x_junction(connection_pts, group_a, group_b, added_lines)    
            except:
                for i in range(len(connection_pts)):
                    added_lines.append(connection_pts[i][0])
                    added_lines.append(avg_pt)
        else:
            need_update = False

    # valence > 4:
    else:
        need_update = True
        if pt_num(junc) == 1 and (len(end) == 0 or manual):
            connect_usm_pts(connection_pts, junc, added_lines)
        elif pt_num(end) >= 1 and pt_num(sharp) >= 1 and pt_num(junc) >=1:
            j = junc if pt_num(junc) == 1 else junc[0]
            connect_usm_pts(connection_pts, j, added_lines)
        elif (pt_num(junc) > 1 or len(end) > 0) and manual == False:
            need_update = False
        elif manual:
            for i in range(len(connection_pts)):
                added_lines.append(connection_pts[i][0])
                added_lines.append(avg_pt)
        else:
            need_update = False
            
    # reset all edge flags in the under sampled region
    if need_update:
        cpt_coord = np.array(np.where(region.astype(bool))).T
        h, w = lines_map_x.shape[0], lines_map_x.shape[1]
        for idx in range(len(cpt_coord)):
            i, j = cpt_coord[idx]
            lines_map_x[i][j] = False
            lines_map_y[i][j] = False
        for k in range(len(connection_pts)):
            i, j, edge = connection_pts[k][2]
            if edge == 2: # bottom
                lines_map_x[i][j] = False
            if edge == 3: # right
                lines_map_y[i][j] = False
            if usm_updated is not None:
                usm_updated[i][j] = True
        assert len(added_lines) > 0
        assert len(added_lines) % 2 == 0
        for i in range(len(added_lines)):
            refined_lines.append(added_lines[i])
    
    return need_update, added_lines

def update_x_junction(connection_pts, group_a, group_b, refined_lines):
    idx0, idx1 = group_a
    idx2, idx3 = group_b
    pt0 = connection_pts[idx0][0]
    pt1 = connection_pts[idx1][0]
    pt2 = connection_pts[idx2][0]
    pt3 = connection_pts[idx3][0]
    done = False
    if (pt0 != pt1).any() and (pt2 != pt3).any(): 
        line1 = Line(complex(*pt0), complex(*pt1))
        line2 = Line(complex(*pt2), complex(*pt3))
        if len(line1.intersect(line2)) == 0:
            refined_lines.append(pt0)
            refined_lines.append(pt1)
            refined_lines.append(pt2)
            refined_lines.append(pt3)
            done = True
    if (pt0 != pt3).any() and (pt2 != pt1).any() and not done: 
        line1 = Line(complex(*pt0), complex(*pt3))
        line2 = Line(complex(*pt2), complex(*pt1))
        if (len(line1.intersect(line2)) == 0):
            refined_lines.append(pt0)
            refined_lines.append(pt3)
            refined_lines.append(pt2)
            refined_lines.append(pt1)

def get_direction_matrix(connection_pts):
    tans = []
    size = len(connection_pts)
    for i in range(size):
        tans.append(connection_pts[i][1])
    dir_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            dir_matrix[i][j] = complex_dot(tans[i], tans[j])
    return dir_matrix

def complex_dot(a, b):
    a_np = np.array((a.real, a.imag))
    b_np = np.array((b.real, b.imag))
    return abs(np.dot(a_np, b_np))

def find_connection_points(pt_grid_map, pt_map, region, under_sampled_map):
    # let's just find the connection points only in this function
    # get region boundary
    # the coordinate in cnt is xy, not ij!
    assert pt_grid_map.shape[0] == pt_map.shape[0] and pt_grid_map.shape[1] == pt_map.shape[1]
    h, w = pt_grid_map.shape[0], pt_grid_map.shape[1]
    # get the coords of region boundary
    cnt, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(cnt) != 1: return False
    cnt = cnt[0].squeeze(axis = 1)
    connection_pts = []
    for idx in range(len(cnt)):
        j, i = cnt[idx]
        if i == 0 or j == 0: continue
        if i > h -5 or j > w - 5: continue # skip if it is too close to right and bottom
        if pt_grid_map[i][j] == False: continue

        # find if there exists poly lines coming from (top, left, bottom, right)
        ef = get_edge_flag_per_grid(pt_grid_map, i, j) # ef is edge flag that indicate if poly line exists
        # find if the (top, left, bottom, right) of current location also has under sampling flag
        # this could be useful to tell if we are on a boundary
        uf = get_under_sampled_flag_per_grid(under_sampled_map, i, j) # usm flag
        # find the direction that has the edge coming and is outside the USM
        cpf = np.logical_and(ef, np.logical_not(uf))# connection points flag
        # connection_pts = [start point, line tangent at break point, coord, where to connect]
        '''
        where to connect:
            0: top
            1: left
            2: bottom
            3: right
        '''
        if cpf.any():
            con_pt_idx = np.where(cpf)[0]
            end_pt = pt_map[i][j] # why we need this?
            if 0 in con_pt_idx:
                start_pt0 = pt_map[i - 1][j]
                connection_pts.append((start_pt0, Line(complex(*start_pt0), complex(*end_pt)).unit_tangent(0), (i-1, j, 0)))
            if 1 in con_pt_idx:
                start_pt1 = pt_map[i][j - 1]
                connection_pts.append((start_pt1, Line(complex(*start_pt1), complex(*end_pt)).unit_tangent(0), (i, j-1, 1)))
            if 2 in con_pt_idx:
                start_pt2 = pt_map[i + 1][j]
                connection_pts.append((start_pt2, Line(complex(*start_pt2), complex(*end_pt)).unit_tangent(0), (i+1, j, 2)))
            if 3 in con_pt_idx:
                start_pt3 = pt_map[i][j + 1]
                connection_pts.append((start_pt3, Line(complex(*start_pt3), complex(*end_pt)).unit_tangent(0), (i, j+1, 3)))

    return connection_pts

def get_edge_flag_per_grid(edge_map, i, j):
    edge0 = edge_map[i-1][j] # top
    edge1 = edge_map[i][j-1] # left
    edge2 = edge_map[i+1][j] # bottom
    edge3 = edge_map[i][j+1] # right
    return np.array([edge0, edge1, edge2, edge3])

def get_under_sampled_flag_per_grid(under_sampled_map, i, j):
    un0 = under_sampled_map[i-1][j] # unsampled neighbour
    un1 = under_sampled_map[i][j-1]
    un2 = under_sampled_map[i+1][j]
    un3 = under_sampled_map[i][j+1]
    return np.array([un0, un1, un2, un3])

def choose_pt(pt_map, idx_to_keypts, idx, thr):
    # get the keypoint and DC grid center point
    dc_pt = pt_map[tuple(idx)]
    
    '''
    if the two point are close to each other enough
    end keypoint probably will be a better choose
    but if there are too faraway, then dual contouring 
    grid center may be better.
    ''' 
    # key_pt = idx_to_keypts[tuple(idx)]
    # dist = np.sqrt(np.square(dc_pt - key_pt).sum())
    # if dist > thr:
    #     return dc_pt
    # else:
    #     return key_pt
    
    return dc_pt

def find_match_keypts(pt_type, idx_to_keypts, val, pt_map, thr = 0.2):
    res = []
    idxs = np.array(np.where(pt_type == val)).T
    if len(idxs) > 1:
        for idx in idxs:
            res.append(choose_pt(pt_map, idx_to_keypts, idx, thr))
        res = np.stack(res, axis = 0)
        # res = res.mean(axis = 0)
    else:
        res = choose_pt(pt_map, idx_to_keypts, idxs[0], thr)
    return np.array(res)

def get_avg_pt(pt_map, pt_region):
    if pt_region.sum() > 0:
        pts = pt_map[np.where(pt_region)].reshape(-1, 2)
        return pts.mean(axis = 0)
    else:
        return []

def query_coord_by_region(region, keypts, key):
    keypt_map = np.zeros(region.shape).astype(bool)
    keypt_coord = keypts[key]
    h, w  = region.shape
    # set overflow coords
    keypt_coord[np.where(keypt_coord[..., 0] >= h)[0], 0] = h - 1
    keypt_coord[np.where(keypt_coord[..., 1] >= w)[0], 1] = w - 1
    keypt_map[tuple(keypt_coord.T)] = True
    keypt = (np.array(np.where(keypt_map & region)).T / 2) 
    keypt[..., (0, 1)] = keypt[..., (1, 0)]
    return keypt.squeeze()

def query_keypt_by_region_no_alginment(keypts, region, pt_map, pt_mask):
    ## re-scale the keypoint coordination and converte them into int coordinations are in (x, y) format
    keypts = copy.deepcopy(keypts) # we don't want to change the input
    for key in keypts:
        keypts[key] = ((keypts[key] + 0.5) * 2).astype(int)
    ## get average points of current region
    region_pt = np.logical_and(region, pt_mask)
    avg_pt = get_avg_pt(pt_map, region_pt)
    
    ## query keypoints covered by region
    end = query_coord_by_region(region, keypts, 'end_point')
    sharp = query_coord_by_region(region, keypts, 'sharp_turn')
    junc = query_coord_by_region(region, keypts, 'junc')
    return np.array(end), np.array(sharp), np.array(junc), avg_pt

def query_keypt_by_region(pt_type, region, idx_to_keypts, pt_map, pt_mask):
    end = []
    sharp = []
    junc = []
    avg_pt = []
    assert region.shape == pt_type.shape
    pt_type = pt_type.copy()
    # make sure region is boolean matrix
    region = region.astype(bool)
    region_pt = np.logical_and(region, pt_mask)
    # set all type info that out of current region to 0 
    pt_type[~region] = 0
    # get query point type
    types = np.unique(pt_type[region])
    # end point
    if 1 in types:
        end = find_match_keypts(pt_type, idx_to_keypts, 1, pt_map)    
    # sharp turn
    if 2 in types:
        sharp = find_match_keypts(pt_type, idx_to_keypts, 2, pt_map)
    # junctions
    if 3 in types:
        junc = find_match_keypts(pt_type, idx_to_keypts, 3, pt_map)
    avg_pt = get_avg_pt(pt_map, region_pt)
    return np.array(end), np.array(sharp), np.array(junc), avg_pt

def lines_to_svg(lines, canvas_w, canvas_h, svg_path = None, indexing = 'xy', paths2Drawing = False):
    # coordinates in lines are y (height), x (width)
    paths_ = Path()
    # assume lines is numpy array
    pt_nums, dims = lines.shape
    # skip if the lines array is not correct
    if dims != 2: return False
    for i in range(0, int(len(lines)//2*2), 2):
        # but svg requires coordinate like x (width), y (height)!
        if indexing == 'ij':
            start = complex(*(lines[i][1], lines[i][0]))
            end = complex(*(lines[i+1][1], lines[i+1][0]))
        elif indexing == "xy":
            start = complex(*(lines[i][0], lines[i][1]))
            end = complex(*(lines[i+1][0], lines[i+1][1]))
        else:
            raise ValueError("Unspported indexing method %s"%indexing)
        if start != end:
            paths_.append(Line(start, end))
    if len(paths_) > 0:
        stroke_widths = [0.5]*len(paths_)
        dimensions = (canvas_w, canvas_h),
        attributes = [{"fill":'none', "stroke":"#000000", "stroke-width":'1', "stroke-linecap":"round"}] * len(paths_)
        if paths2Drawing:
            return wsvg(
                paths_, 
                stroke_widths = stroke_widths, 
                dimensions = (canvas_w, canvas_h), 
                filename = svg_path, 
                attributes = attributes,
                paths2Drawing = True).tostring()
        else:
            assert svg_path is not None
            wsvg(
                paths_, 
                stroke_widths = stroke_widths, 
                dimensions = (canvas_w, canvas_h),
                attributes = attributes, 
                filename = svg_path)
    
    if path.exists(svg_path):    
        return True
    else:
        return False

def roll_edge(edge, shift, axis):
    assert len(edge.shape) == 2
    assert shift != 0
    edge_ = edge.copy().astype(bool)
    if axis == 'x':
        if shift > 0:
            edge_[-shift:, :] = False
        else:
            edge_[:-shift, :] = False
        return np.roll(edge_, shift = shift, axis = 0)
    elif axis == 'y':
        if shift > 0:
            edge_[:, -shift:] = False
        else:
            edge_[:, :-shift] = False
        return np.roll(edge_, shift = shift, axis = 1)
    else:
        raise ValueError("axis could only be x or y but got %s"%axis)

def logical_minus(a, b):
    assert a.shape == b.shape
    assert a.dtype == bool
    assert b.dtype == bool
    return a^b&a

def map_to_lines(edge_map, keypt_map, tensor_input = False, refine = False):
    '''
    import pickle
    with open("map_to_lines.pickle","wb") as f: pickle.dump( ( edge_map, keypt_map, tensor_input, refine ),  f )
    with open('map_to_lines_test.py','w') as f: print( """from utils.ndc_tools import map_to_lines
import pickle
args = pickle.load( open('map_to_lines.pickle','rb') )
import time
start = time.time()
map_to_lines( *args )
print( time.time() - start, "seconds" )
""", file = f )
    print( "map_to_lines_test.py" )
    '''
    
    '''
    Given:
        edge_map:   A boolean array with shape (h, w, 2)
        keypt_map:  A float array with shape (h, w, 2)
    Return:
        lines:          A float array with shape (k, 2), stores the start and end point of each
                        line reconstruction in a compact way
        lines_map_x:    A float array with shape (h, w , 2), stores the lines intersect with each
                        grid edge along x-axis in a sparse way
        lines_map_y:    A float array with shape (h, w , 2), stores the lines intersect with each
                        grid edge along y-axis in a sparse way
    '''
    if tensor_input:
        edge_map = edge_map.squeeze().detach().to(bool).cpu().numpy()
        keypt_map = keypt_map.squeeze().detach().cpu().numpy()
    h, w, _= edge_map.shape
    # remove tiny lines in the edge map, DON'T use this flag when training the network!
    if refine:
        edge_map = edge_map.astype(bool)
        # cut small branch
        edge_map = cut_branch(edge_map)
        skel = remove_stray(edge_map[..., 0]|edge_map[..., 1])
        '''
            below function are still experimental, and may still buggy
        '''
        # remove repeat (adjacent) strokes
        skel = edge_map[..., 0] | edge_map[..., 1]
        skel = remove_parallel(skel, edge_map)
        edge_map = fill_edge_hole(edge_map & skel[..., np.newaxis])
        edge_map = cut_branch(edge_map & skel[..., np.newaxis])
        

    assert edge_map.shape[0] == keypt_map.shape[0] and edge_map.shape[1] == keypt_map.shape[1]
    
    '''
    lines_map_x = np.zeros((h, w)).astype(object)
    lines_map_y = np.zeros((h, w)).astype(object)
    # it is not possible to record dual contouring lines along y-axis at the 1st row
    # and also not possbile to record dc lines along x-axis at the 1st column
    for i in range(h):
        for j in range(w):
            if edge_map[i][j][0] and i != 0: # edge map x
                if (keypt_map[i - 1][j] == keypt_map[i][j]).all() == False:
                    s = keypt_map[i - 1][j]
                    e = keypt_map[i][j]
                    # if (s > 1e-2).all() or (e > 1e-2).all():
                    #     lines_map_x[i][j] = (s, e)
                    lines_map_x[i][j] = (s, e)
            if edge_map[i][j][1] and j != 0: # edge map y
                if (keypt_map[i][j - 1] == keypt_map[i][j]).all() == False:
                    s = keypt_map[i][j - 1]
                    e = keypt_map[i][j]
                    # if (s > 1e-2).all() or (e > 1e-2).all():
                    #     lines_map_y[i][j] = (s, e)
                    lines_map_y[i][j] = (s, e)
    
    return linemap_to_lines(lines_map_x, lines_map_y), lines_map_x, lines_map_y, edge_map
    '''
    
    # Get all the True i,j locations.
    # We don't want the first row/column, so skip them in the `where` and offset the resulting row/column indices.
    wherex_left = np.where( edge_map[1:,:,0] )
    wherex_right = ( wherex_left[0] + 1, wherex_left[1] )
    wherey_up = np.where( edge_map[:,1:,1] )
    wherey_down = ( wherey_up[0], wherey_up[1] + 1 )
    
    # Allocate space for the line-segments x 2 (start and end) x 2 (x and y) data
    line_segments = np.zeros( ( len(wherex_left[0]) + len(wherey_up[0]), 2, 2 ), dtype = keypt_map.dtype )
    # Fill it.
    line_segments[ :len(wherex_left[0]), 0, : ] = keypt_map[ wherex_left ]
    line_segments[ :len(wherex_left[0]), 1, : ] = keypt_map[ wherex_right ]
    line_segments[ len(wherex_left[0]):, 0, : ] = keypt_map[ wherey_up ]
    line_segments[ len(wherex_left[0]):, 1, : ] = keypt_map[ wherey_down ]
    # Extract zero-length segments
    line_segments = line_segments[ ( line_segments[:,0,:] != line_segments[:,1,:] ).any( axis = 1 ) ]
    lines = line_segments.reshape( -1, 2 )
    # assert ( lines == linemap_to_lines(lines_map_x, lines_map_y) ).all()
    
    # Make 2D object array from the data.
    # This would be better as a np.ma.MaskedArray, but there are a lot of places in the code to change.
    lines_map_x2 = np.zeros((h, w)).astype(object)
    lines_map_y2 = np.zeros((h, w)).astype(object)
    xmask = ( keypt_map[ wherex_left ] != keypt_map[ wherex_right ] ).any( axis = 1 )
    lines_map_x2[ wherex_right[0][xmask], wherex_right[1][xmask] ] = list(zip( keypt_map[ wherex_left ][xmask], keypt_map[ wherex_right ][xmask] ))
    ymask = ( keypt_map[ wherey_up ] != keypt_map[ wherey_down ] ).any( axis = 1 )
    lines_map_y2[ wherey_down[0][ymask], wherey_down[1][ymask] ] = list(zip( keypt_map[ wherey_up ][ymask], keypt_map[ wherey_down ][ymask] ))
    # assert ( np.where(lines_map_x)[0] == np.where(lines_map_x2)[0] ).all()
    # assert ( np.where(lines_map_x)[1] == np.where(lines_map_x2)[1] ).all()
    # assert ( np.where(lines_map_y)[0] == np.where(lines_map_y2)[0] ).all()
    # assert ( np.where(lines_map_y)[1] == np.where(lines_map_y2)[1] ).all()
    # assert ( lines_map_y[ np.where(lines_map_y) ][0] == lines_map_y2[ np.where(lines_map_y2) ][0] ).all()
    # assert ( np.array( lines_map_x[ np.where(lines_map_x) ][0] ) == np.array( lines_map_x2[ np.where(lines_map_x) ][0] ) ).all()
    # assert ( np.array( lines_map_x[ np.where(lines_map_x) ][1] ) == np.array( lines_map_x2[ np.where(lines_map_x) ][1] ) ).all()
    # assert ( np.array( lines_map_y[ np.where(lines_map_y) ][0] ) == np.array( lines_map_y2[ np.where(lines_map_y) ][0] ) ).all()
    # assert ( np.array( lines_map_y[ np.where(lines_map_y) ][1] ) == np.array( lines_map_y2[ np.where(lines_map_y) ][1] ) ).all()
    lines_map_x = lines_map_x2
    lines_map_y = lines_map_y2
    
    return lines, lines_map_x, lines_map_y, edge_map

def remove_stray(skel, thr = 6):
    _, edge_regions = cv2.connectedComponents(skel.astype(np.uint8), connectivity = 8)
    mask = np.zeros(skel.shape).astype(bool)
    for r in np.unique(edge_regions):
        if r == 0: continue
        m = edge_regions == r
        if m.sum() < thr: mask[m] = True
    skel = skel & ~mask
    return skel

def remove_parallel(skel, edge_map, stray_thr = 6):
    skel = skeletonize(skel)
    # remove stray short lines 
    skel = remove_stray(skel, stray_thr)
    # fill 8-way connection of -45 degree diagonal to 4 way conncection
    skel_4way = logical_minus(roll_edge(skel, -1, 'x'), skel) & roll_edge(skel, 1, 'y')
    skel = skel | skel_4way
    skel = np.repeat(skel[..., np.newaxis], 2, axis = -1)
    # the rest edge labels could still exist 8-way connections, need to check this again
    skel_ = edge_map & skel
    skel_ = skel_[..., 0] | skel_[..., 1]
    skel_4way = logical_minus(roll_edge(skel_, -1, 'x'), skel_)
    skel_4way = roll_edge(roll_edge(skel_4way, -1, 'y'), 1, 'x') & roll_edge(skel_, 1, 'x')
    skel = skel_ | skel_4way
    return skel

def cut_branch(edge_map):
    edge_x = edge_map[..., 0]
    edge_y = edge_map[..., 1]
    cross_k = np.array([[0,1,0],[1,1,1],[0,1,0]]).astype(np.uint8)
    val_map, _ = get_valence_np(edge_x, edge_y)
    end_map = val_map == 1
    junc_map = val_map > 2
    branch_map = convolve2d(end_map.astype(float), cross_k, mode = 'same', boundary = 'fill', fillvalue = 0) > 0
    branch_map = branch_map & junc_map
    branch_map = cv2.dilate(branch_map.astype(np.uint8), cross_k, iterations = 1 ) > 0
    branch_map = branch_map & end_map
    edge_x_ = logical_minus(edge_x, branch_map)
    edge_y_ = logical_minus(edge_y, branch_map)
    branch_map_ = logical_minus(branch_map, edge_x | edge_y)
    edge_x = logical_minus(edge_x_, roll_edge(branch_map_, 1, 'x'))
    edge_y = logical_minus(edge_y_, roll_edge(branch_map_, 1, 'y'))
    return np.stack((edge_x, edge_y), axis = -1)

def fill_edge_hole(edge_map):
    edge_x = edge_map[..., 0]
    edge_y = edge_map[..., 1]
    # fill holes in matched lines
    kernel_x_hole = np.array([[-1, 1, -1], [-1, -1, -1], [-1, 1, -1]])
    kernel_y_hole = np.array([[-1, -1, -1], [1, -1, 1], [-1, -1, -1]])
    hole_x = convolve2d(edge_x, kernel_x_hole, mode = 'same', boundary = 'fill', fillvalue = 0) == 2
    hole_y = convolve2d(edge_y, kernel_y_hole, mode = 'same', boundary = 'fill', fillvalue = 0) == 2
    edge_x = edge_x | hole_x
    edge_y = edge_y | hole_y
    '''
    fill holes in mismatched lines:
    case1:
    -------
           -------
    -------
         -------
    case2:
    |
    |
      |
      |

    |
    | |
    | |
      |
   
    '''

    kernel_xy1_hole = np.array([[1, -1, 0], [1, -1, 0], [-1, 1, 0]])
    kernel_xy2_hole = np.array([[-1, 1, 0], [-1, 1, 0], [1, -1, 0]])
    kernel_yx1_hole = np.array([[1, 1, -1], [-1, -1, 1], [0, 0, 0]])
    kernel_yx2_hole = np.array([[-1, -1, 1], [1, 1, -1], [0, 0, 0]])
    kernel_yx1_overlap = np.array([[1, 1, -1], [-1, 1, 1], [0, 0, 0]])
    kernel_yx2_overlap = np.array([[-1, -1, 1], [1, 1, 1], [0, 0, 0]])
    kernel_xy1_overlap = np.array([[1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    kernel_xy2_overlap = np.array([[-1, 1, 0], [-1, 1, 0], [1, 1, 0]])
    hole_y1 = correlate(edge_x.astype(int), kernel_xy1_hole, mode = 'constant') == 3
    hole_y2 = correlate(edge_x.astype(int), kernel_xy2_hole, mode = 'constant') == 3
    hole_y3 = correlate(edge_x.astype(int), kernel_xy1_overlap, mode = 'constant') == 4
    hole_y4 = correlate(edge_x.astype(int), kernel_xy2_overlap, mode = 'constant') == 4
    hole_x1 = correlate(edge_y.astype(int), kernel_yx1_hole, mode = 'constant') == 3
    hole_x2 = correlate(edge_y.astype(int), kernel_yx2_hole, mode = 'constant') == 3
    hole_x3 = correlate(edge_y.astype(int), kernel_yx1_overlap, mode = 'constant') == 4
    hole_x4 = correlate(edge_y.astype(int), kernel_yx2_overlap, mode = 'constant') == 4
    edge_x = logical_minus( edge_x | hole_x1 | hole_x2 | hole_x3 | hole_x4, roll_edge(hole_y4, 1, 'x') | hole_y3)
    edge_y = logical_minus( edge_y | hole_y1 | hole_y2 | hole_y3 | hole_y4, roll_edge(hole_x4, 1, 'y') | hole_x3)
    # remove connection between adjacent 2 t-junctions
    val_map, _ = get_valence_np(edge_map[..., 0], edge_map[..., 1])
    kernel_tt_x = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
    kernel_tt_y = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
    branch_x = correlate((val_map >= 3).astype(int), kernel_tt_x, mode = 'constant') == 2
    branch_y = correlate((val_map >= 3).astype(int), kernel_tt_y, mode = 'constant') == 2
    edge_x = logical_minus(edge_x, branch_x)
    edge_y = logical_minus(edge_y, branch_y)
    return np.stack((edge_x, edge_y), axis = -1)

def linemap_to_lines(lines_map_x, lines_map_y):
    lines_x = lines_map_x[np.where(lines_map_x != False)]
    lines = []
    if len(lines_x) > 0:
        lines_x = np.concatenate(lines_x)
        lines.append(lines_x)
    lines_y = lines_map_y[np.where(lines_map_y != False)]
    if len(lines_y) > 0:
        lines_y = np.concatenate(lines_y)
        lines.append(lines_y)
    if len(lines) == 2:
        lines = np.concatenate((lines_x, lines_y), axis = 0)
    elif len(lines) ==1:
        lines = lines[0]
    else:
        raise ValueError("irregular line maps!")
    return lines.reshape(-1, 2)
