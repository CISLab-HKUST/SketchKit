from sketchkit.core.sketch import Sketch, Path, Curve, Point
from typing import List, Dict, Any
import numpy as np
from .compute_cost import cost_individual, cost_transition, compute_T_junctions
from operator import itemgetter
from .tsp_bnb import Node
from queue import PriorityQueue
from itertools import count
from .direction import determine_direction
from sketchkit.renderer.cairo_renderer import CairoRenderer
from sketchkit.ordering.orderer import Orderer
from tqdm import tqdm
from PIL import Image

__all__ = ["AnimatedDrawer"]





class AnimatedDrawer(Orderer):
    """
    Implementation of Animated construction of line drawings.
    
    This method analyzes stroke relationships, computes transition costs between strokes,
    and determines the optimal drawing order using a branch-and-bound algorithm.
    """
    
    def __init__(
        self,
        max_k: int = 4,
        w: float = 0.1111111,
        canvas_size: int = 800,
        canvas_color=(1, 1, 1),
        width: int = 4,
    ):
        """
        Initialize the AnimatedDrawer with the specified parameters.
        
        Args:
            max_k: Maximum number of closest strokes to consider for each stroke.
            w: Weight parameter for combining proximity and continuity costs.
            canvas_size: Size of the canvas for rendering (if used).
            canvas_color: Color of the canvas for rendering (if used).
            width: Width of the strokes for rendering (if used).
        """
        self.max_k = max_k
        self.w = w
        self.canvas_size = canvas_size
        self.canvas_color = canvas_color
        self.width = width

    def run(self, sketch: Sketch) -> Sketch:
        """
        Run the animated drawing algorithm to determine the optimal stroke order.
        
        Args:
            sketch: Sketch object containing the static sketch to animate.
        
        Returns:
            Sketch: A Sketch object with ordered paths, where the order of paths corresponds to the optimized drawing sequence.
        """
        result = self._animated_drawing(sketch)
        paths = [sketch.paths[i] for i in result["solution"]]
        # Empty paths have no ordering cost, but must not shift stroke indices.
        paths.extend(path for path in sketch.paths if not path.curves)
        ret_sketch = Sketch(height=sketch.height, width=sketch.width, paths=paths,
                            attributes=sketch.attributes)
        # sketch_frames = self._render_animated_drawing(sketch, result)
        # renderer = CairoRenderer(canvas_size=self.canvas_size, canvas_color=self.canvas_color)
        # frames = []
        # for sketch in tqdm(sketch_frames, desc="Rendering frames"):
        #     frame = renderer.render(sketch, width=self.width)
        #     frames.append(Image.fromarray(frame))

        # # Save as animated GIF
        # frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

        return ret_sketch

    def _get_cost_bi(self, cost_bi, i, j):
        """
        Get the cost between two strokes from a bidirectional cost matrix.
        
        Args:
            cost_bi: Bidirectional cost matrix.
            i: Index of the first stroke.
            j: Index of the second stroke.
        
        Returns:
            float: The cost between the two strokes.
        """
        if i == j:
            return 0
        else:
            ii = min(i, j)
            jj = max(i, j)
            return cost_bi[ii][jj - ii - 1]

    def _animated_drawing(self, sketch: Sketch) -> Dict[str, Any]:
        """
        Create an animated drawing from a static sketch.

        Args:
            sketch: The static sketch to animate.
        
        Returns:
            Dict: A dictionary containing the optimal solution (stroke order) and directions.
        """
        max_k = self.max_k
        w = self.w
        strokes_txy = []
        cost_uni = []
        path_indices = []

        for path_index, path in enumerate(sketch.paths):
            if path.curves is None or len(path.curves) < 1:
                continue
            # print([[[curve.p_start.x, curve.p_start.y], [curve.p_end.x, curve.p_end.y]] for curve in path.curves])
            polyline_path = []
            for curve in path.curves:
                if len(polyline_path) == 0:
                    polyline_path.append(0)
                    polyline_path.append(curve.p_start.x)
                    polyline_path.append(curve.p_start.y)

                elif (
                    np.linalg.norm(
                        np.array(polyline_path[-2:])
                        - np.array([curve.p_start.x, curve.p_start.y])
                    )
                    > 1
                ):
                    polyline_path.append(0)
                    polyline_path.append(curve.p_start.x)
                    polyline_path.append(curve.p_start.y)

            if (
                np.linalg.norm(
                    np.array(polyline_path[-2:])
                    - np.array([path.curves[-1].p_end.x, path.curves[-1].p_end.y])
                )
                > 1
            ):
                polyline_path.append(0)
                polyline_path.append(path.curves[-1].p_end.x)
                polyline_path.append(path.curves[-1].p_end.y)
            strokes_txy.append(polyline_path)
            path_indices.append(path_index)
            cost = cost_individual(polyline_path)
            cost_uni.append(cost)

        cost_bi_pro = []
        cost_bi_col = []

        for sid1 in range(len(strokes_txy) - 1):
            cost_bi_pro_row = []
            cost_bi_col_row = []
            for sid2 in range(sid1 + 1, len(strokes_txy)):
                pro, col = cost_transition(strokes_txy[sid1], strokes_txy[sid2])
                cost_bi_pro_row.append(pro)
                cost_bi_col_row.append(col)
            cost_bi_pro.append(cost_bi_pro_row)
            cost_bi_col.append(cost_bi_col_row)

        T_junctions = []
        for sid1 in range(len(strokes_txy)):
            for sid2 in range(len(strokes_txy)):
                if sid1 != sid2:
                    c = compute_T_junctions(strokes_txy[sid1], strokes_txy[sid2])
                    if c > 0:
                        T_junctions.append([sid1, sid2])

        vertex_pair = []
        edge_weight = []
        n = len(cost_uni)
        if n == 0:
            return {"solution": [], "cost": 0, "directions": []}
        for i in range(n):
            all_pro_for_i = []
            for j in range(n):
                if j != i:
                    all_pro_for_i.append(self._get_cost_bi(cost_bi_pro, i, j))
                else:
                    all_pro_for_i.append(999999999)
            for k in range(min(max_k, n - 1)):
                # get the index of the k-th closest stroke
                j = sorted(enumerate(all_pro_for_i), key=itemgetter(1))[k][0]
                if [i, j] in T_junctions:
                    # add j, i
                    vertex_pair.append((j, i))
                    edge_weight.append(
                        w * self._get_cost_bi(cost_bi_pro, j, i)
                        + (1 - w) * self._get_cost_bi(cost_bi_col, j, i)
                    )
                elif [j, i] in T_junctions:
                    # add i, j
                    vertex_pair.append((i, j))
                    edge_weight.append(
                        w * self._get_cost_bi(cost_bi_pro, i, j)
                        + (1 - w) * self._get_cost_bi(cost_bi_col, i, j)
                    )
                else:
                    # add j, i
                    vertex_pair.append((j, i))
                    edge_weight.append(
                        w * self._get_cost_bi(cost_bi_pro, j, i)
                        + (1 - w) * self._get_cost_bi(cost_bi_col, j, i)
                    )
                    # add i, j
                    vertex_pair.append((i, j))
                    edge_weight.append(
                        w * self._get_cost_bi(cost_bi_pro, i, j)
                        + (1 - w) * self._get_cost_bi(cost_bi_col, i, j)
                    )
                ### NOTE didn't check conflicting directed cycles, could use toposort to check
        weight_binary = {}
        for i in range(len(vertex_pair)):
            weight_binary[str(vertex_pair[i][0]) + "," + str(vertex_pair[i][1])] = (
                edge_weight[i]
            )
        weight_unary = cost_uni

        optim_node = None
        upper_bound = float("inf")
        q = PriorityQueue()
        sequence = count()
        # Directed T-junction constraints may require a start other than stroke 0.
        for start in range(n):
            root = Node([], start, weight_unary, weight_binary, n)
            lower_bound = root.lower_bound()
            if lower_bound[0]:
                q.put((root.cost + lower_bound[1], next(sequence), root))

        print_count = 0

        while not q.empty():
            this_node = q.get()[2]
            if len(this_node.trace) == n - 1:
                if this_node.cost < upper_bound:
                    upper_bound = this_node.cost
                    optim_node = this_node
            else:
                next_nodes = []
                for key in weight_binary.keys():
                    if (
                        int(key.split(",")[0]) == this_node.curr
                        and int(key.split(",")[1]) not in this_node.trace
                    ):
                        next_nodes.append(int(key.split(",")[1]))
                for next_node in next_nodes:
                    next_node_q = Node(
                        trace=this_node.trace + [this_node.curr],
                        curr=next_node,
                        weight_unary=weight_unary,
                        weight_binary=weight_binary,
                        n=n,
                    )
                    next_lower_bound = next_node_q.lower_bound()
                    if (
                        next_lower_bound[0]
                        and next_node_q.cost + next_lower_bound[1] <= upper_bound
                    ):
                        q.put((next_node_q.cost + next_lower_bound[1],
                               next(sequence), next_node_q))
            if print_count % 10000 == 0:
                print(this_node.trace + [this_node.curr], upper_bound)
            print_count = print_count + 1

        if optim_node is None:
            raise ValueError(
                "No complete stroke order satisfies the Fu neighbour graph and "
                "T-junction constraints. Try increasing max_k."
            )
        else:
            result = {
                "solution": [path_indices[i] for i in optim_node.trace + [optim_node.curr]],
                "cost": optim_node.cost,
            }
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

    # def _render_animated_drawing(
    #     self, sketch: Sketch, result: dict, sample_rate: int = 24
    # ):
    #     """
    #     Render the animated drawing based on the result.
    #     Args:
    #         sketch (Sketch): The sketch object.
    #         result (dict): The result dictionary.
    #     """
    #     sketch_frames = []
    #     # 将sketch.paths按照result["solution"]的顺序排序
    #     paths = [sketch.paths[i] for i in result["solution"]]
    #     # paths = [path.remove_empty_curves() for path in paths]

    #     # 在0-1之间均匀取100个点
    #     ts = np.linspace(0, 1, sample_rate)

    #     for i in range(len(result["solution"]) - 1):
    #         base_pathes = [] if i == 0 else paths[:i]
    #         index = result["solution"][i]
    #         path_points = sketch.paths[index].get_division_points(ts)
    #         # print("path_points:", path_points)
    #         path_points = path_points if result["directions"][i] else path_points[::-1]
    #         curves = []
    #         for j in range(len(path_points) - 1):
    #             curves.append(Curve.from_line([path_points[j], path_points[j + 1]]))
    #         for k in range(len(curves)):
    #             # base_pathes.append(Path(curves[:k+1]))
    #             # print("base_pathes:", base_pathes[-1].curves[-1].p_end.x, base_pathes[-1].curves[-1].p_end.y)
    #             sketch_frames.append(Sketch(base_pathes + [Path(curves[: k + 1])]))

    #     return sketch_frames
