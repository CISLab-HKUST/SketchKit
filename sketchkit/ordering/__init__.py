from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from typing import List
import numpy as np
import sketchkit.ordering.compute_cost as compute_cost
from operator import itemgetter
from sketchkit.ordering.tsp_bnb import Node
from queue import PriorityQueue
from sketchkit.ordering.direction import determine_direction
from sketchkit.utils.transform import line_to_cubic
from sketchkit.renderer.cairo_renderer import CairoRenderer
max_k = 4
w = 1/9

def get_cost_bi(cost_bi, i, j):
    if i == j:
        return 0
    else:
        ii = min(i, j)
        jj = max(i, j)
        return cost_bi[ii][jj - ii - 1]

def animated_drawing(sketch: Sketch) -> List[int]:
    """
    Create an animated drawing from a static sketch.

    Args:
        sketch (Sketch): The static sketch to animate.
    Returns:
        List[int]: A list of path indices for the animated sketch.
    """

    strokes_txy = []
    cost_uni = []

    for path in sketch.paths:
        if path.curves is None or len(path.curves) < 1:
            continue
        # print([[[curve.p_start.x, curve.p_start.y], [curve.p_end.x, curve.p_end.y]] for curve in path.curves])
        polyline_path = []
        for curve in path.curves:
            if len(polyline_path) == 0:
                polyline_path.append(0)
                polyline_path.append(curve.p_start.x)
                polyline_path.append(curve.p_start.y)
                
            elif np.linalg.norm(np.array(polyline_path[-2:]) - np.array([curve.p_start.x, curve.p_start.y])) > 1:
                polyline_path.append(0)
                polyline_path.append(curve.p_start.x)
                polyline_path.append(curve.p_start.y)
                
        if np.linalg.norm(np.array(polyline_path[-2:]) - np.array([path.curves[-1].p_end.x, path.curves[-1].p_end.y])) > 1:
            polyline_path.append(0)
            polyline_path.append(path.curves[-1].p_end.x)
            polyline_path.append(path.curves[-1].p_end.y)
        strokes_txy.append(polyline_path)
        cost = compute_cost.cost_individual(polyline_path)
        cost_uni.append(cost)

    cost_bi_pro = []
    cost_bi_col = []

    for sid1 in range(len(strokes_txy) - 1):
        cost_bi_pro_row = []
        cost_bi_col_row = []
        for sid2 in range(sid1 + 1, len(strokes_txy)):
            pro, col = compute_cost.cost_transition(strokes_txy[sid1], strokes_txy[sid2])
            cost_bi_pro_row.append(pro)
            cost_bi_col_row.append(col)
        cost_bi_pro.append(cost_bi_pro_row)
        cost_bi_col.append(cost_bi_col_row)

    T_junctions = []
    for sid1 in range(len(strokes_txy)):
        for sid2 in range(len(strokes_txy)):
            if sid1 != sid2:
                c = compute_cost.compute_T_junctions(strokes_txy[sid1], strokes_txy[sid2])
                if c > 0:
                    T_junctions.append([sid1, sid2])

    vertex_pair = []
    edge_weight = []
    n = len(cost_uni)
    for i in range(n):
        if n < max_k:
            continue
        all_pro_for_i = []
        for j in range(n):
            if j != i:
                all_pro_for_i.append(get_cost_bi(cost_bi_pro, i, j))
            else:
                all_pro_for_i.append(999999999)
        for k in range(max_k):
            # get the index of the k-th closest stroke
            j = sorted(enumerate(all_pro_for_i), key=itemgetter(1))[k][0]
            if [i, j] in T_junctions:
                # add j, i
                vertex_pair.append((j, i))
                edge_weight.append(w * get_cost_bi(cost_bi_pro, j, i) + (1-w) * get_cost_bi(cost_bi_col, j, i))
            elif [j, i] in T_junctions:
                # add i, j
                vertex_pair.append((i, j))
                edge_weight.append(w * get_cost_bi(cost_bi_pro, i, j) + (1-w) * get_cost_bi(cost_bi_col, i, j))
            else:
                # add j, i
                vertex_pair.append((j, i))
                edge_weight.append(w * get_cost_bi(cost_bi_pro, j, i) + (1-w) * get_cost_bi(cost_bi_col, j, i))
                # add i, j
                vertex_pair.append((i, j))
                edge_weight.append(w * get_cost_bi(cost_bi_pro, i, j) + (1-w) * get_cost_bi(cost_bi_col, i, j))
            ### NOTE didn't check conflicting directed cycles, could use toposort to check
    weight_binary = {}
    for i in range(len(vertex_pair)):
        weight_binary[str(vertex_pair[i][0]) + "," + str(vertex_pair[i][1])] = edge_weight[i]
    weight_unary = cost_uni
    
    optim_node = None
    upper_bound = 999999999
    root = Node(trace=[], curr=0, weight_unary=weight_unary, weight_binary=weight_binary, n=n)
    q = PriorityQueue()
    q.put((root.cost + root.lower_bound()[1], root))

    print_count = 0

    while not q.empty():
        this_node = q.get()[1]
        if len(this_node.trace) == n - 1:
            if this_node.cost < upper_bound:
                upper_bound = this_node.cost
                optim_node = this_node
        else:
            next_nodes = []
            for key in weight_binary.keys():
                if int(key.split(",")[0]) == this_node.curr and \
                    int(key.split(",")[1]) not in this_node.trace:
                    next_nodes.append(int(key.split(",")[1]))
            for next_node in next_nodes:
                next_node_q = Node(trace=this_node.trace + [this_node.curr], curr=next_node, weight_unary=weight_unary, weight_binary=weight_binary, n=n)
                next_lower_bound = next_node_q.lower_bound()
                if next_lower_bound[0] and \
                    next_node_q.cost + next_lower_bound[1] <= upper_bound:
                    q.put((next_node_q.cost + next_lower_bound[1], next_node_q))
        if print_count % 10000 == 0:
            print(this_node.trace + [this_node.curr], upper_bound)
        print_count = print_count + 1

    if optim_node is None:
        print("Solution does not exist!")
        return None
    else:
        result = {"solution": optim_node.trace + [optim_node.curr], "cost": optim_node.cost}
        # print("Optimal solution:", optim_node.trace + [optim_node.curr], "cost:", optim_node.cost)
        # return result["solution"]  
        directions = []
        for i in range(len(result["solution"])):
            index = result["solution"][i]
            if i == 0:
                prev = None
            elif directions[-1]:
                prev = sketch.paths[result["solution"][i - 1]].curves[-1].p_end
            else:
                prev = sketch.paths[result["solution"][i - 1]].curves[0].p_start
            current_path = sketch.paths[index]
            start = current_path.curves[0].p_start
            end = current_path.curves[-1].p_end
            
            directions.append(determine_direction(prev, start, end))
        result["directions"] = directions
        print(result)
        return result
    
def render_animated_drawing(sketch: Sketch, result: dict):
    """
    Render the animated drawing based on the result.
    Args:
        sketch (Sketch): The sketch object.
        result (dict): The result dictionary.
    """
    sketch_frames = []
    # 将sketch.paths按照result["solution"]的顺序排序
    paths = [sketch.paths[i] for i in result["solution"]]
    # paths = [path.remove_zero_length_curves() for path in paths]
    
    # 在0-1之间均匀取100个点
    ts = np.linspace(0, 1, 24)
    
    for i in range(len(result["solution"])-1):
        base_pathes = [] if i == 0 else paths[:i]
        index = result["solution"][i]
        path_points = sketch.paths[index].get_division_points(ts)
        # print("path_points:", path_points)
        path_points = path_points if result["directions"][i] else path_points[::-1]
        curves = []
        for j in range(len(path_points) - 1):
            curves.append(line_to_cubic([path_points[j], path_points[j+1]]))
        for k in range(len(curves)):
            # base_pathes.append(Path(curves[:k+1]))
            # print("base_pathes:", base_pathes[-1].curves[-1].p_end.x, base_pathes[-1].curves[-1].p_end.y)
            sketch_frames.append(Sketch(base_pathes+[Path(curves[:k+1])]))
    
    return sketch_frames

        
          


    