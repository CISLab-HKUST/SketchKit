from pathlib import Path as P
import sys
directory = P(__file__)
if directory not in sys.path:
    d1 = str(directory.parent)
    if d1 not in sys.path:
        sys.path.append(d1)
    d2 = str(directory.parent.parent)
    if d2 not in sys.path:
        sys.path.append(str(directory.parent.parent))
import numpy as np
import time
import xml.etree.ElementTree as ET
import copy


from os.path import split
from svgpathtools import Path, Line, svg2paths
from svgpathtools.document import flattened_paths
# https://stackoverflow.com/questions/30447355/speed-of-k-nearest-neighbour-build-search-with-scikit-learn-and-scipy
# we should use KDTree in sklearn since it is implemented by Cython
# from scipy.spatial import KDTree
from random import sample

def open_svg_flatten(path_to_svg):
    filepath, svg = split(path_to_svg)

    # the svgpathtools has one function for flatting the paths but it only support xml root as the input
    # hopefully this could remove all transformations
    doc = ET.parse(path_to_svg)
    root = doc.getroot()

    # get paths with base coordinates (transforms applied)
    try:
        flatpaths = flattened_paths(root)
    except:
        flatpaths, _ = svg2paths(path_to_svg)
    paths = []
    
    # remove nested path structrue
    for p in flatpaths:
        for s in p:
            paths.append(s)

    # clean up invalid path
    for i in range(len(paths) - 1, -1, -1):
        if paths[i].start == None or paths[i].end == None:         
            paths.pop(i)
        elif paths[i].length() == 0: 
            paths.pop(i)

    # read canvas size
    try:
        h = int(root.attrib['height'].strip("px"))
        w = int(root.attrib['width'].strip("px"))
    except:
        h = int(float(root.attrib['viewBox'].split(" ")[3].strip("px")))
        w = int(float(root.attrib['viewBox'].split(" ")[2].strip("px")))

    return paths, (h, w)

def dot(A, B):
    return A.real * B.real + A.imag * B.imag

def flatten_paths_to_pts_array(paths, unique = False):
    all_pts = np.asarray(paths)
    start_pts = np.stack((all_pts[:, 0].real, all_pts[:, 0].imag), axis = 1)
    end_pts = np.stack((all_pts[:, 1].real, all_pts[:, 1].imag), axis = 1)
    assert start_pts.shape == end_pts.shape
    lines_array = np.concatenate((start_pts, end_pts), axis = 1)
    pts_array = lines_array.reshape(-1, 2)
    if unique:
        # we can't simply use np.unique to cleanup redundant points cause this 
        # function will also re-order the remained points.
        map_to_clean = [0]
        for i in range(1, len(pts_array)):
            if (pts_array[i - 1] == pts_array[i]).all(): continue
            map_to_clean.append(i)
        map_to_clean = np.array(map_to_clean)
        pts_array = pts_array[map_to_clean]
    return pts_array

def build_edge(pts_array, inv = None):
    ## Yotam: numpy.unique() can replace this entire function.
    ## I don't have a way to test the `inv` parameter, but I believe it can be handled with `return_index = True`.
    if inv is None:
        unique_pts, edges = np.unique( pts_array, axis = 0, return_inverse = True )
        edges = edges.reshape(-1,2)
        return edges, unique_pts
        # edges2 = edges

    # initial edge graphs
    assert len(pts_array) % 2 == 0
    edges = np.arange(len(pts_array)).reshape(-1, 2)
    # edges = np.insert((np.arange(2, len(pts_array)+1)//2), 0, 0).reshape(-1, 2)
    visited_pts = []
    zero_dist = 1e-5
    for i in range(len(pts_array)):
        pt = pts_array[i]
        if (pt == -1).all(): continue
        dist = np.sqrt((pts_array - pt[np.newaxis, :])**2).sum(axis = -1)
        # search connected points
        cpt_idx = np.where(dist < zero_dist)[0]
        cpt_idx = np.delete(cpt_idx, np.where(cpt_idx == i))
        # if found connections 
        if len(cpt_idx) > 0 and i not in visited_pts:
            # remove duplicate connected points
            pts_array[cpt_idx] = -1
            for e in cpt_idx:
                edges[edges == e] = i
        visited_pts.append(i)
    
    # cleanup graphs, we could let the code run faster if skip this step
    pt_idx_old_to_new = np.ones(len(pts_array)).astype(int) * -1
    new_idx_counter = 0
    for i in range(len(pts_array)):
        if (pts_array[i] != -1).any():
            pt_idx_old_to_new[i] = new_idx_counter
            new_idx_counter += 1
    mask = pts_array != -1
    pts_array = pts_array[mask].reshape(-1, 2)
    edges = pt_idx_old_to_new[edges]
    if inv is not None:
        assert (mask[:, 0] == mask[:, 1]).all()
        return edges, pts_array, inv[mask[:, 0]]
    return edges, pts_array

def build_connectivity(pts_array, edges):
    cn_dict = {} # connected neighbour dictionary
    fedge = edges.flatten()
    ptidx = np.arange(len(fedge))
    valence = np.zeros(len(pts_array))
    for i in range(len(pts_array)):
        if (pts_array[i] == -1).all(): continue
        # find the idx of the point which is identical to the ith pt
        idp = np.where(fedge == i)[0]
        # find the idx of the point which is the neighbour of the ith pt
        sp_mask = idp % 2 == 0
        ep_mask = idp % 2 == 1
        inn = idp.copy()
        if sp_mask.any():
            inn[sp_mask] = fedge[inn[sp_mask] + 1]
        if ep_mask.any():
            inn[ep_mask] = fedge[inn[ep_mask] - 1]
        inn = np.unique(inn)
        # update the connectivity dict and valence list
        if len(inn) == 0:
            cn_dict[i] = None
        else:
            cn_dict[i] = inn
            valence[i] = len(inn)
    return cn_dict, valence

def update_unvisited_pt(valence, unvisited_pts, idx):
    assert valence[idx] >= 0
    if valence[idx] == 0:
        return False
    else:
        valence[idx] = valence[idx] - 1
        if valence[idx] == 0: unvisited_pts.remove(idx)
        return True

def compute_dist_M(pt1, pt2):
    return np.sqrt(((pt1[:,np.newaxis, :] - pt2[np.newaxis, :, :])**2).sum(axis = -1).astype(float))

def find_alignment(pts_array, val_pt_idx, keypt, val, valence, pt_type, thres = 1.5):
    '''
    Given:
        pts_array, N x 2 array, all DC vertices
        val_pt_idx, M x 1 array, indices of all DC vertices that has the same valence, the number of valence is defined by val
        keypt, K x 2 array, detected key points we want to aligned with.
        val, a int, indicating which type of the DC vertices and keypoints we want to align
        valence, N x 1 array, the valence of each DC vertices
        pt_type, N x 1 array, point alginment result and this variable will be inplace updated, therefore there is NO return results
        thres, a float, distance threshold for point alignment
    Action:
        align all predicted key point to its nearest DC vertex if the distance between them is less than thres, otherwise drop the keypoint as false positive.
    '''
    # do nothing if no dual contouring vertices or keypoint is found
    if len(val_pt_idx[0]) == 0 or len(keypt)== 0: return None
    
    # compute distance matrix between to point groups
    M_dist = compute_dist_M(pts_array[val_pt_idx], keypt)
    
    # for each DC vertex, find the closest keypoint
    M_idx = np.argsort(M_dist, axis = 0)
    
    # if the distance is less than given threshold, then we find a sucessful alginment
    aligned_idx = np.where(M_dist[M_idx[0,:], np.arange(M_dist.shape[1])] <= thres)
    found_idx = M_idx[0,:][aligned_idx]

    # update result
    for idx in found_idx:
        # # if the valence is 2, check if the current vertex is really a sharp turn
        # if val == 2:
        #     edges = connectivity[val_pt_idx[0][idx]]
        #     # compute the dot product of two segment's unit tangnet
        #     ang = dot(Line(start = complex(*pts_array[edges[0]]), end = complex(*pts_array[val_pt_idx[0][idx]])).unit_tangent(), 
        #         Line(start = complex(*pts_array[val_pt_idx[0][idx]]), end = complex(*pts_array[edges[1]])).unit_tangent())
        #     if ang > 0.6: continue
        # # if multiple key points are allocated to a small range 
        pt_type[val_pt_idx[0][idx]] = val
    return None

def find_alignment_all(keypts, valence, pts_array, connectivity, pt_type = None, thre_v1 = 1.5, thre_v2 = 1.5, thre_vx = 5, sec_pass = False):
    if pt_type is None:
        pt_type = np.ones(len(pts_array)) * - 1
    v0_pt_idx = np.where(valence == 0)
    v1_pt_idx = np.where(valence == 1)
    v2_pt_idx = np.where(valence == 2)
    v3_pt_idx = np.where(valence >= 3)
    end_pt = keypts['end_point']
    sharp_pt = keypts['sharp_turn']
    junc_pt = keypts['junc']
    # end point
    find_alignment(pts_array, v1_pt_idx, end_pt, 1, valence, pt_type, thres = thre_v1)
    # sharp turn
    find_alignment(pts_array, v2_pt_idx, sharp_pt, 2, valence, pt_type, thres = thre_v2)
    # high valence junction
    if sec_pass == False:
        find_alignment(pts_array, v3_pt_idx, junc_pt, 3, valence, pt_type, thres = thre_vx)

    v1 = pts_array[np.where(pt_type == 1)]
    v1 = [complex(*pt) for pt in v1]
    v2 = pts_array[np.where(pt_type == 2)]
    v2 = [complex(*pt) for pt in v2]
    if sec_pass == False:
        v3 = pts_array[np.where(pt_type >= 3)]
        v3 = [complex(*pt) for pt in v3]
        vx = v1+v2+v3
    else:
        vx = v1+v2

    return vx, pt_type, pts_array[np.where(pt_type >= 3)]

def update_valence(path_by_idx, valence, connectivity, unvisited_edges, visited_vertex, end_pts, st_pts, loop_clean = False):
    for i in range(len(path_by_idx)):
        p = path_by_idx[i]
        v_new = 0
        remove = True
        for next_p in connectivity[p]:
            if set([p, next_p]) in unvisited_edges: 
                v_new += 1
                remove = False
        if remove:
            visited_vertex[p] = True
            if p in end_pts: end_pts.remove(p)
            if p in st_pts: st_pts.remove(p)
        valence[p] = v_new

    if loop_clean:
        for n in connectivity[path_by_idx[0]]:
            if valence[n] > 0:
                valence[n] = valence[n] - 1
            else:
                ed = set([n, path_by_idx[-1]])
                if ed in unvisited_edges: unvisited_edges.remove(ed)

def bfs_all_pairs(s, ends, connectivity, pts_array, visited_vertex):
    # prepare bfs search
    min_len = float("inf")
    min_e = None
    s_shortest = None
    min_pidx = None
    for e in ends:
        stroke, pidx = bfs_single_pair(s, e, connectivity, pts_array, visited_vertex)
        if stroke is None: continue # skip if no connection found
        if len(stroke) < min_len:
            s_shortest = stroke
            min_len = len(stroke)
            min_e = e
            min_pidx = pidx
    return s_shortest, min_e, min_pidx

def bfs_single_pair(s, e, neighbours, pts_array, visited):
    def solve(s, neighbours, n, visited):
        # s, idx for start point
        # neighbours, dictionary, neighours of each vertex
        # n, number of vetices of the graph
        q = []
        q.append(s)
        visited = visited.copy()
        visited[s] = True
        prev = (np.ones(n) * -1).astype(int)

        while len(q)!=0:
            node = q.pop()
            nghbs = neighbours[node]
            if len(nghbs) > 0:
                for next_node in nghbs:
                    if visited[next_node]: continue
                    q.append(next_node)
                    visited[next_node] = True
                    prev[next_node] = node
        return prev

    

    def recon_path(s, e, prev, pts_array):
        res = Path()
        at = e
        p = []
        while prev[at] != -1:
            p.append(at)
            spt = complex(*pts_array[at])
            ept = complex(*pts_array[prev[at]])
            res.append(Line(start = spt, end = ept))
            at = prev[at]
        p.append(at)
        if p[-1] == s:
            p.reverse()
            return res.reversed(), p
        else:
            return None, None

    prev = solve(s, neighbours, len(pts_array), visited)
    if prev is not None:
        res, p = recon_path(s, e, prev, pts_array)
        return res, p
    else:
        return None, None

def simplify_graph(paths, keypts = None, mode = 'hybird', skip_len = 4):

    pts_array = flatten_paths_to_pts_array(paths)
    edges, pts_array = build_edge(pts_array)
    connectivity, valence = build_connectivity(pts_array, edges)
    valence_ = valence.copy()
    unvisited_pts = set(edges.flatten())

    # align keypoints onto the strokes
    # types definition: not keypoint(-1), end point(0), sharp turn(1), T junction(2), X junction(3), Star junction(4)
    pt_type = np.ones(len(pts_array)) * - 1
    nodes = None
    if keypts is not None:
        nodes = find_alignment_all(keypts, valence, pts_array, connectivity, pt_type, thre_v1 = 1.5, thre_v2 = 1.5, thre_vx = 5)

    def traversal_shortest_path(pts_array, pt_type, unvisited_edges, connectivity, valence):
        # collect all end points
        st_pts = set(np.where(pt_type == 1)[0])
        end_pts = set(np.where(pt_type > 1)[0])
        v1_pts = set(np.where(valence == 1)[0])
        # random pick one end point
        # compute all shortest path to other end points, select the shortest one among them
        # remove all vertices invovled from graph, repeat this process until all end point has been used up
        res = []
        visited_vertex = (np.zeros(len(pts_array))).astype(bool)
        imp_s_e = []
        cc = 0
        bfs_time = 0
        update_time = 0
        while len(st_pts) > 0 or len(v1_pts) > 1:
            # pick up the start point
            if len(st_pts) > 0:
                s = sample(sorted(st_pts), 1)[0]
                assert valence[s] == 1
            else:
                s = sample(sorted(v1_pts), 1)[0]

            # prepare the end point candidates
            ends = st_pts.union(end_pts) - set([s])
            v1s = v1_pts - set([s]) - st_pts
            
            # 1st pass, find shortest path between start point to each keypoint
            start_time = time.time()
            s_shortest, min_e, min_pidx = bfs_all_pairs(s, ends, connectivity, pts_array, visited_vertex)
                
            # 2nd pass, find shortest path between start point to all valence 1 vertices
            if s_shortest is None:
                s_shortest, min_e, min_pidx = bfs_all_pairs(s, v1s, connectivity, pts_array, visited_vertex)
            end_time = time.time()
            bfs_time = bfs_time + (end_time - start_time)

            # record result if find one shortest path
            start_time = time.time()
            if s_shortest is not None:
                might_loop = True
                # remove edges, first test if we could safely remove all edges
                for i in range(len(min_pidx) - 1):
                    eg = set([min_pidx[i], min_pidx[i+1]])
                    if eg in unvisited_edges:
                        unvisited_edges.remove(eg)
                        might_loop = False
                    # update valence and visited_vertex
                    update_valence(min_pidx, valence, connectivity, unvisited_edges, visited_vertex, end_pts, st_pts)
                    if might_loop:
                        print("log:\tfind loop, remove the start point and restart the reaserching")
                        for n in connectivity[s]:
                            ed = set([s, n])
                            if ed in unvisited_edges:unvisited_edges.remove(ed)
                        update_valence([s], valence, connectivity, unvisited_edges, visited_vertex, end_pts, st_pts, True)
                        assert s not in end_pts
                        assert s not in st_pts
                if len(s_shortest) > 1:     
                    res.append(s_shortest)
            else:
                # this is a dirty fix, but I still don't know why no path found is possible
                update_valence([s], valence, connectivity, unvisited_edges, visited_vertex, end_pts, st_pts, True)
            end_time = time.time()
            update_time = update_time + (end_time - start_time)
            v1_pts = set(np.where(valence == 1)[0])
            
        print("\nlog:\tbfd time %ds, update time %ds"%(bfs_time, update_time))
        assert (valence >= 0).all()
        # assert len(unvisited_edges) == 0 or len(unvisited_edges) == 1
        return res

    if mode.lower() == 'bfs':
        edges = [set(e) for e in edges]
        strokes = traversal_shortest_path(pts_array, pt_type, edges, connectivity, valence)
    elif mode.lower() == 'hybird':
        ## reconstruct stroke topology
        def find_next_hop_graph(cur_pt_idx, pre_pt_idx, unvisited_pts, connectivity, valence, res_stroke, pts_array, pt_type, dfs = False, pre_dfs = False):
            def find_next_hop_dfs(cur_pt_idx, pre_pt_idx, unvisited_pts, connectivity, res_stroke, pts_array, valence, pre_dfs):
                # if pt_type[cur_pt_idx] > 1:
                #     return cur_pt_idx, pre_pt_idx, False, False
                # find tangent of current stroke
                pre_tan = res_stroke[-1].unit_tangent()
                max_dot = -2
                max_tan_idx = None
                next_stroke = [] # stroe all strokes in next hop
                # we don't search backward
                next_pt = set(connectivity[cur_pt_idx]).intersection(unvisited_pts) - set([pre_pt_idx])
                if len(next_pt) == 0:
                    return cur_pt_idx, pre_pt_idx, False, False
                if len(next_pt) == 1:
                    if pre_dfs:
                        # if reaches to the end point
                        npt = next_pt.pop()
                        res_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[npt])))
                        update_unvisited_pt(valence, unvisited_pts, npt)
                        return npt, cur_pt_idx, False, False
                    else:
                        npt = next_pt.pop()
                        res_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[npt])))
                        update_unvisited_pt(valence, unvisited_pts, npt)
                        return npt, cur_pt_idx, True, True
                for i in next_pt:
                    next_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[i])))
                    cur_tan = next_stroke[-1].unit_tangent()
                    cur_dot = dot(cur_tan, pre_tan)
                    if max_dot < cur_dot:
                        max_dot = cur_dot
                        max_tan_idx = i
                res_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[max_tan_idx])))
                update_unvisited_pt(valence, unvisited_pts, max_tan_idx)
                return max_tan_idx, cur_pt_idx, True, False

            def add_pt_to_removal(pt_idxs, valence):
                pt_idxs_ = copy.deepcopy(pt_idxs)
                for idx in pt_idxs:
                    if valence[idx] > 0:
                        valence[idx] = valence[idx] - 1
                        if valence[idx] > 0:
                            pt_idxs_.remove(idx)
                    else:
                        # if we found the current pt's valence is 0 already
                        # then we should skip this point
                        assert valence[idx] == 0
                        pt_idxs_.remove(idx)
                return pt_idxs_

            def sync_val(old, new):
                assert (new >= 0).all()
                assert len(old) == len(new)
                # for i in range(len(old)): old[i] = new[i]
                old[:] = new[:]
                assert (old == new).all()

            # if pt_type[cur_pt_idx] >= 1:
            #     return cur_pt_idx, pre_pt_idx, False, False
            # if dfs flag is true, run dfs mode directly
            if dfs:
                if update_unvisited_pt(valence, unvisited_pts, cur_pt_idx):
                    return find_next_hop_dfs(cur_pt_idx, pre_pt_idx, unvisited_pts, connectivity, res_stroke, pts_array, valence, pre_dfs)
                else:
                    return cur_pt_idx, pre_pt_idx, False, False
            # else start bfs mode
            nodes = [set(), set(), set()]
            wbrpt = set() # will be removed points
            ## bfs search lv1
            valence_ = valence.copy()
            neighbours = set(connectivity[cur_pt_idx]).intersection(unvisited_pts) - set([pre_pt_idx])
            nodes[0] = neighbours
            if len(neighbours) == 0:
                # if reaches to the endpoint, return directly
                return cur_pt_idx, pre_pt_idx, False, False
            else:
                wbrpt = add_pt_to_removal(neighbours, valence_)
                for pt in neighbours: 
                    # if pt_type[pt] > 2:
                    #     return pt, cur_pt_idx, False, False       
                    wbrpt = wbrpt.union(add_pt_to_removal(set([cur_pt_idx]), valence_))
            ## bfs search lv2
            rm_lv1 = []
            loop1_ = []
            for pt in nodes[0]:
                # if pt in wbrpt: continue
                # neighbours at lv2
                neighbours = set(connectivity[pt]).intersection(unvisited_pts - wbrpt) - set([cur_pt_idx, pre_pt_idx])
                loop1_.append(set(connectivity[pt]).intersection(unvisited_pts) - set([cur_pt_idx, pre_pt_idx]))
                wbrpt = wbrpt.union(add_pt_to_removal(neighbours, valence_))
                if len(neighbours) == 0:
                    rm_lv1.append(pt)
                    continue
                for pt_ in neighbours:
                    ptn = add_pt_to_removal([pt], valence_)
                    wbrpt = wbrpt.union(ptn)
                    nodes[1].add(pt_)
            if len(rm_lv1) > 0 and len(nodes[1]) > 0:
                for i in rm_lv1: nodes[0].remove(i)
            # check if inner loop case 1 is detected
            if len(loop1_) > 1:
                loop1 = loop1_[0]
                for i in range(2, len(loop1_)):
                    loop1 = loop1.intersection(loop1_[i])
            else:
                loop1 = set()
            # check if inner loop case 2 is detected
            loop2 = nodes[0].intersection(nodes[1])
            if len(loop2) == 1:
                nodes[0] = loop2
                nodes[1] = nodes[1] - loop2
            ## tell if there matches any cases that could be proceed.
            if DEBUG:
                import pdb
                pdb.set_trace()
            if len(nodes[0]) == 1 and len(nodes[1]) == 1:
                n0 = nodes[0].pop()
                n1 = nodes[1].pop()
                res_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[n0])))
                res_stroke.append(Line(start = complex(*pts_array[n0]), end = complex(*pts_array[n1])))
                for pt in wbrpt: unvisited_pts.remove(pt)
                sync_val(valence, valence_)
                return n1, n0, False, False
            elif len(nodes[0]) == 1 and len(nodes[1]) == 0:
                n0 = nodes[0].pop()
                res_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[n0])))
                for pt in wbrpt: unvisited_pts.remove(pt)
                sync_val(valence, valence_)
                return n0, cur_pt_idx, False, False
            elif len(nodes[0]) > 1 and len(nodes[1]) > 1:
                if update_unvisited_pt(valence, unvisited_pts, cur_pt_idx):
                    return find_next_hop_dfs(cur_pt_idx, pre_pt_idx, unvisited_pts, connectivity, res_stroke, pts_array, valence, pre_dfs)
                else:
                    # break searh here and start a new search if next two levels contains branches
                    return cur_pt_idx, pre_pt_idx, False, False    
                # return cur_pt_idx, pre_pt_idx, False, False
            
            ## bfs search lv3
            rm_lv2 = []
            for pt in nodes[1]:
                # if pt in wbrpt: continue
                neighbours = set(connectivity[pt]).intersection(unvisited_pts - wbrpt)  - nodes[0] - set([cur_pt_idx, pre_pt_idx])
                wbrpt = wbrpt.union(add_pt_to_removal(neighbours, valence_))
                if len(neighbours) == 0:
                    rm_lv2.append(pt)
                    continue
                for pt_ in neighbours:
                    ptn = add_pt_to_removal([pt], valence_)
                    wbrpt = wbrpt.union(ptn)
                    nodes[2].add(pt_)
            if len(rm_lv2) > 0 and len(nodes[2]) > 0:
                for i in rm_lv2: nodes[1].remove(i)
            if DEBUG:
                import pdb
                pdb.set_trace()
            # if find inner loop between lv0 and lv2, build up the stroke and return new index for next hoop
            if len(nodes[0]) >= 1 and len(nodes[1]) == 1 and len(nodes[2]) == 1:
                n0 = nodes[0].pop()
                n1 = nodes[1].pop()
                n2 = nodes[2].pop()
                res_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[n0])))
                res_stroke.append(Line(start = complex(*pts_array[n0]), end = complex(*pts_array[n1])))
                res_stroke.append(Line(start = complex(*pts_array[n1]), end = complex(*pts_array[n2])))
                for pt in wbrpt: unvisited_pts.remove(pt)
                sync_val(valence, valence_)
                return n2, n1, False, False
            # if find multiple branch, fall back to dfs search
            elif len(nodes[0]) > 1:
                if update_unvisited_pt(valence, unvisited_pts, cur_pt_idx):
                    return find_next_hop_dfs(cur_pt_idx, pre_pt_idx, unvisited_pts, connectivity, res_stroke, pts_array, valence, pre_dfs)
                else:
                    return cur_pt_idx, pre_pt_idx, False, False
            # if find branch at next 2 hop, move to the next hop only
            elif len(nodes[0]) == 1 and (len(nodes[1]) > 1 or len(nodes[2]) > 1):
                n0 = nodes[0].pop()
                res_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[n0])))
                update_unvisited_pt(valence, unvisited_pts, n0)
                update_unvisited_pt(valence, unvisited_pts, cur_pt_idx)
                return n0, cur_pt_idx, False, False
            else:
                return cur_pt_idx, pre_pt_idx, False, False

        DEBUG = False
        first_iter = False
        strokes = []
        # while there still exists unallocated pts
        while len(unvisited_pts) != 0:
            ed = set(np.where(pt_type == 1)[0]).intersection(unvisited_pts)
            if len(ed) > 0 and DEBUG == False:
                sp = sample(sorted(ed), 1)[0]
            else:
                sp = sample(sorted(unvisited_pts), 1)[0]
            if DEBUG and first_iter:
                sp = 19
                first_iter = False
            if DEBUG: print("log:\tsq = %d"%sp)
            eps = set(connectivity[sp]).intersection(unvisited_pts)
            assert(update_unvisited_pt(valence, unvisited_pts, sp))
            # if we hit an endpoint, then skip half of the searching
            skip_sp = False
            if len(eps) == 1:
                skip_sp = True
            if len(eps) == 0:
                if (valence == 0).all():
                    break
                else:
                    continue    
            ep = eps.pop()
            if DEBUG and first_iter:
                ep = 17
            if DEBUG: print("log:\teq = %d"%ep)
            assert(update_unvisited_pt(valence, unvisited_pts, ep))
            cur_pidx = sp
            pre_pidx = ep
            if skip_sp == False:
                res_stroke = Path(Line(start = complex(*pts_array[sp]), end = complex(*pts_array[ep])).reversed())
            else:
                res_stroke = Path(Line(start = complex(*pts_array[sp]), end = complex(*pts_array[ep])))
            dfs = False
            pre_dfs = False
            # search along start point
            if skip_sp == False:
                while True:
                    # maybe need to rewrite the logic here, like summerize everything into a function
                    cur_pidx_new, pre_pidx_new, dfs, pre_dfs = find_next_hop_graph(
                        cur_pidx, 
                        pre_pidx, 
                        unvisited_pts, 
                        connectivity, 
                        valence, 
                        res_stroke, 
                        pts_array,
                        pt_type,
                        dfs, 
                        pre_dfs)
                    if cur_pidx_new != cur_pidx or pre_pidx_new != pre_pidx:
                        pre_pidx = pre_pidx_new
                        cur_pidx = cur_pidx_new
                    else:
                        break
                res_stroke = res_stroke.reversed()
            
            # search along end point
            cur_pidx = ep
            pre_pidx = sp
            dfs = False
            pre_dfs = False
            while True and len(unvisited_pts) > 0:
                cur_pidx_new, pre_pidx_new, dfs, pre_dfs = find_next_hop_graph(
                    cur_pidx, 
                    pre_pidx, 
                    unvisited_pts, 
                    connectivity, 
                    valence, 
                    res_stroke, 
                    pts_array, 
                    pt_type,
                    dfs, 
                    pre_dfs)
                if cur_pidx_new != cur_pidx or pre_pidx_new != pre_pidx:
                    pre_pidx = pre_pidx_new
                    cur_pidx = cur_pidx_new
                else:
                    break

            # there could have inner loop which length greater than 1
            if len(res_stroke) > skip_len: 
                strokes.append(res_stroke)
            # strokes.append(res_stroke)
    else:
        raise ValueError("unsupported running mode, the mode should only be one in the set (bfs, hybird)")
    return strokes, nodes

def ramerDouglas(strokes, epsilon):
    from rdp import rdp
    # Ramer-Douglas-Peucker simplify
    res = []
    for s in strokes:
        sim = rdp(flatten_paths_to_pts_array(s, True), epsilon = epsilon)
        p_new = Path()
        for i in range(1, len(sim)):
            p_new.append(Line(start = complex(*sim[i-1]), end = complex(*sim[i])))
        res.append(p_new)
    return res


