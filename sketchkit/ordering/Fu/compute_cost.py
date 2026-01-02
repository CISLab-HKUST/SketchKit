import cv2
import json
import numpy as np
import sys
from math import sqrt, acos, pi

def remove_same_xy(txy):
    txy_new = [txy[0], txy[1], txy[2]]
    for vid in range(1, len(txy) // 3):
        if txy[3*vid+1] != txy_new[-2] or txy[3*vid+2] != txy_new[-1]:
            txy_new.append(txy[3*vid])
            txy_new.append(txy[3*vid+1])
            txy_new.append(txy[3*vid+2])
    return txy_new

def stroke_length(txy):
    l = 0
    for vid in range(1, len(txy) // 3):
        l = l + sqrt((np.float32(txy[3*vid+1]) - np.float32(txy[3*vid-2])) ** 2 + (np.float32(txy[3*vid+2]) - np.float32(txy[3*vid-1])) ** 2)
    return l

def curvatures(txy):
    if len(txy) // 3 < 9:
        return [0]
    ks = []
    # https://en.wikipedia.org/wiki/Finite_difference_coefficient
    coeff_d1 = [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]
    coeff_d2 = [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]
    for vid in range(4, len(txy) // 3 - 4):
        xs = [np.float32(txy[3*vid_+1]) for vid_ in range(vid-4, vid+5)]
        ys = [np.float32(txy[3*vid_+2]) for vid_ in range(vid-4, vid+5)]
        d1x = 0
        d1y = 0
        d2x = 0
        d2y = 0
        for i in range(9):
            d1x = d1x + xs[i] * coeff_d1[i]
            d1y = d1y + ys[i] * coeff_d1[i]
            d2x = d2x + xs[i] * coeff_d2[i]
            d2y = d2y + ys[i] * coeff_d2[i]
        k = abs(d1x * d2y - d2x * d1y) / ((d1x ** 2 + d1y ** 2) ** 1.5)
        # k in [0, 10] covers 99.21% of all curvatures
        # k in [0, 5] covers 97.86% of all curvatures
        # k in [0, 2] covers 94.51% of all curvatures, here 2 is chosen to limit the range of std(ks)
        ks.append(k if k < 2 else 2)
    return ks

def cost_individual(txy):
    # NOTE ita==0.1 in the original paper, here ita==1 so two terms are of comparable scale
    ita = 1
    dist_start_end = sqrt((np.float32(txy[1]) - np.float32(txy[-2])) ** 2 + (np.float32(txy[2]) - np.float32(txy[-1])) ** 2)
    deviation_from_straight = ita * (1 - dist_start_end / stroke_length(txy))
    ks = curvatures(txy)
    deviation_from_circle = np.std(ks)
    # deviation_from_straight in [0, 1], majority in [0, 0.1]
    # deviation_from_circle in [0, 1], majority in [0, 0.1]
    return deviation_from_straight + deviation_from_circle

def dist_closest_points(txy1, txy2):
    closest_dist = 800 * sqrt(2)
    for vid1 in range(len(txy1) // 3):
        for vid2 in range(len(txy2) // 3):
            curr_dist = sqrt((np.float32(txy1[3*vid1+1]) - np.float32(txy2[3*vid2+1])) ** 2 + (np.float32(txy1[3*vid1+2]) - np.float32(txy2[3*vid2+2])) ** 2)
            closest_dist = curr_dist if curr_dist < closest_dist else closest_dist
    return closest_dist

def compute_thetas(txy1, txy2, is_begin1, is_begin2):
    # line too short, quit
    if len(txy1) // 3 < 5 or len(txy2) // 3 < 5:
        return 0, 0
    end_point_x1 = np.float32(txy1[1]) if is_begin1 else np.float32(txy1[-2])
    end_point_y1 = np.float32(txy1[2]) if is_begin1 else np.float32(txy1[-1])
    near_point_x1 = np.float32(txy1[3*4+1]) if is_begin1 else np.float32(txy1[-3*4-2])
    near_point_y1 = np.float32(txy1[3*4+2]) if is_begin1 else np.float32(txy1[-3*4-1])
    tangent_vec1 = np.asarray([end_point_x1 - near_point_x1, end_point_y1 - near_point_y1])
    end_point_x2 = np.float32(txy2[1]) if is_begin2 else np.float32(txy2[-2])
    end_point_y2 = np.float32(txy2[2]) if is_begin2 else np.float32(txy2[-1])
    near_point_x2 = np.float32(txy2[3*4+1]) if is_begin2 else np.float32(txy2[-3*4-2])
    near_point_y2 = np.float32(txy2[3*4+2]) if is_begin2 else np.float32(txy2[-3*4-1])
    tangent_vec2 = np.asarray([end_point_x2 - near_point_x2, end_point_y2 - near_point_y2])
    vector_from1to2 = np.asarray([end_point_x2 - end_point_x1, end_point_y2 - end_point_y1])
    # tangent line too short, quit
    if np.linalg.norm(tangent_vec1) == 0 or np.linalg.norm(tangent_vec2) == 0:
        return 0, 0
    assert np.linalg.norm(vector_from1to2) > 0
    cos1 = np.sum(tangent_vec1 * vector_from1to2) / (np.linalg.norm(tangent_vec1) * np.linalg.norm(vector_from1to2))
    cos2 = np.sum(tangent_vec1 * (-vector_from1to2)) / (np.linalg.norm(tangent_vec1) * np.linalg.norm(vector_from1to2))
    # force round to avoid numerical errors
    cos1 = 1 if cos1 > 1 else cos1
    cos1 = -1 if cos1 < -1 else cos1
    cos2 = 1 if cos2 > 1 else cos1
    cos2 = -1 if cos2 < -1 else cos1
    return acos(cos1), acos(cos2)

def process_end_points(txy1, txy2):
    b1b2 = sqrt((np.float32(txy1[1]) - np.float32(txy2[1])) ** 2 + (np.float32(txy1[2]) - np.float32(txy2[2])) ** 2)
    b1e2 = sqrt((np.float32(txy1[1]) - np.float32(txy2[-2])) ** 2 + (np.float32(txy1[2]) - np.float32(txy2[-1])) ** 2)
    e1b2 = sqrt((np.float32(txy1[-2]) - np.float32(txy2[1])) ** 2 + (np.float32(txy1[-1]) - np.float32(txy2[2])) ** 2)
    e1e2 = sqrt((np.float32(txy1[-2]) - np.float32(txy2[-2])) ** 2 + (np.float32(txy1[-1]) - np.float32(txy2[-1])) ** 2)
    gap = min(b1b2, b1e2, e1b2, e1e2)
    if gap == 0:
        return gap, 0, 0
    if gap == b1b2:
        theta1, theta2 = compute_thetas(txy1, txy2, is_begin1=True, is_begin2=True)
    elif gap == b1e2:
        theta1, theta2 = compute_thetas(txy1, txy2, is_begin1=True, is_begin2=False)
    elif gap == e1b2:
        theta1, theta2 = compute_thetas(txy1, txy2, is_begin1=False, is_begin2=True)
    else:
        theta1, theta2 = compute_thetas(txy1, txy2, is_begin1=False, is_begin2=False)
    return gap, theta1, theta2

def cost_transition(txy1, txy2):
    # wp = 1/9
    proximity = dist_closest_points(txy1, txy2) / (800 * sqrt(2))
    gap, theta1, theta2 = process_end_points(txy1, txy2)
    collinearity = gap / (stroke_length(txy1) + stroke_length(txy2)) * (theta1 + theta2) ** 2
    # over 98% collinearity values are in [0, 100]
    collinearity = 100 if collinearity > 100 else collinearity
    collinearity = collinearity / 100
    return proximity, collinearity

def on_segment(p1, p2, p3):
    # check if p2 is on line segment p1 to p3
    if p2[0] <= max(p1[0], p3[0]) and p2[0] >= min(p1[0], p3[0]) and p2[1] <= max(p1[1], p3[1]) and p2[1] >= min(p1[1], p3[1]):
        return True
    return False

def orientation(p1, p2, p3):
    val = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])
    if val == 0:
        return 0 # collinear
    return 1 if val > 0 else 2 # clock or counterclock wise

def line_segments_intersect(p1, p2, q1, q2):
    # p1 to p2 is a line segment, q1 to q2 is another line segment
    # general case
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        cos_angle = np.abs(np.sum((p2 - p1) * (q2 - q1))) / (np.linalg.norm(p2 - p1) * np.linalg.norm(q2 - q1))
        return True, acos(cos_angle)
    # special case
    if o1 == 0 and on_segment(p1, q1, p2):
        return True, 0
    if o2 == 0 and on_segment(p1, q2, p2):
        return True, 0
    if o3 == 0 and on_segment(q1, p1, q2):
        return True, 0
    if o4 == 0 and on_segment(q1, p2, q2):
        return True, 0
    return False, 0

def compute_T_junctions(txy1, txy2):
    # asymmetric, txy1 substrate, txy2 attachment
    # line too short, quit
    if len(txy1) // 3 < 5 or len(txy2) // 3 < 5:
        return 0
    start_point = np.asarray([np.float32(txy2[1]), np.float32(txy2[2])])
    start_tangent1 = np.asarray([np.float32(txy2[3*4+1]), np.float32(txy2[3*4+2])])
    start_tangent2 = 2 * start_point - start_tangent1
    end_point = np.asarray([np.float32(txy2[-2]), np.float32(txy2[-1])])
    end_tangent1 = np.asarray([np.float32(txy2[-3*4-2]), np.float32(txy2[-3*4-1])])
    end_tangent2 = 2 * end_point - end_tangent1
    # check if (vertex1, vertex2) intersects (tangent1, tangent2)
    # https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    is_detected_start = False
    for sid in range(len(txy1) // 3 - 1):
        vertex1 = np.asarray([np.float32(txy1[3*sid+1]), np.float32(txy1[3*sid+2])])
        vertex2 = np.asarray([np.float32(txy1[3*sid+4]), np.float32(txy1[3*sid+5])])
        is_intersect_start, angle_start = line_segments_intersect(vertex1, vertex2, start_tangent1, start_tangent2)
        if is_intersect_start and angle_start > 20 / 180 * pi:
            confidence = min(stroke_length(txy1[:3*sid+3]), stroke_length(txy1[3*sid+3:])) / stroke_length(txy1)
            if confidence > 0.05:
                is_detected_start = True
                break
    is_detected_end = False
    for sid in range(len(txy1) // 3 - 1):
        vertex1 = np.asarray([np.float32(txy1[3*sid+1]), np.float32(txy1[3*sid+2])])
        vertex2 = np.asarray([np.float32(txy1[3*sid+4]), np.float32(txy1[3*sid+5])])
        is_intersect_end, angle_end = line_segments_intersect(vertex1, vertex2, end_tangent1, end_tangent2)
        if is_intersect_end and angle_end > 20 / 180 * pi:
            confidence = min(stroke_length(txy1[:3*sid+3]), stroke_length(txy1[3*sid+3:])) / stroke_length(txy1)
            if confidence > 0.05:
                is_detected_end = True
                break
    return (is_detected_start << 1) + is_detected_end
