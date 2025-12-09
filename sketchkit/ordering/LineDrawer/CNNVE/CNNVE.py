"""
Main animator class for LineDrawer - video-synchronized sketch animation
"""

import os
import glob
import pickle
import gzip
import cv2
import numpy as np

from .video_processor import VideoProcessor
from .frame_matcher import FrameMatcher  
from .color_generator import ColorGenerator
from sketchkit.vectorization.LineDrawer.CNNVE.utils.image_utils import criterion_mask_overlap, get_rect_p0p1
from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point

# MASK_TH is defined in utils.py as 50, but we can define it here or import it if it exists in vectorization utils
# In vectorization/utils/image_utils.py, criterion_mask_overlap uses hardcoded 30 for I_th and merge_th argument
# Let's define MASK_TH here for consistency with previous animation code
MASK_TH = 50


class CNNVE:
    """
    LineDrawer animation generator that synchronizes strokes with video frames.
    Renamed to CNNVE to match vectorization structure.
    
    Attributes:
        indir (str): Input directory containing .graph_merge files and videos
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the LineDrawer animator.
        
        Args:
            device (str, optional): Device to use (cpu/cuda). Defaults to "cpu".
        """
        self.device = device
        self.video_processor = VideoProcessor()
        self.frame_matcher = FrameMatcher()
        self.color_generator = ColorGenerator()
    
    def run(self, img: np.ndarray, video_path: str) -> list[Sketch]:
        """
        Generate animation sketches from image and video.
        
        Args:
            img (np.ndarray): Input image as a 2D numpy array (grayscale).
            video_path (str): Path to video file for frame synchronization.
            
        Returns:
            list[Sketch]: List of generated Sketch objects for each frame
        """
        import tempfile
        from PIL import Image
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_img_path = os.path.join(temp_dir, "input.png")
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            Image.fromarray(img).save(temp_img_path)
            
            from sketchkit.vectorization.LineDrawer.CNNVE import CNNVE as VectorizationCNNVE
            vectorizer = VectorizationCNNVE(device=self.device)
            vectorizer.run(temp_img_path)
            
            graph_merge_path = os.path.join(temp_dir, "input.graph_merge")
            
            if not os.path.exists(graph_merge_path):
                raise RuntimeError(f"CNNVE failed to generate graph file at {graph_merge_path}")
            
            with gzip.open(graph_merge_path, 'rb') as f:
                merge_dic = pickle.load(f)
            
            img_data = cv2.imread(temp_img_path, 0)
            
            frames = self._vis_merge_result_paints_undo(
                img_data, merge_dic, video_path
            )
            
            return frames

    
    def _vis_merge_result_paints_undo(self, im_ori_c1, merge_dic, video_path):
        """
        Visualize stroke merging with video synchronization.
        
        Args:
            im_ori_c1: Original image
            merge_dic: Merge dictionary with stroke data
            video_path: Path to video file
            
        Returns:
            list[Sketch]: List of generated Sketch objects
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        ref_frames = self.video_processor.mp4_to_imgs(video_path, False, im_ori_c1.copy())
        print(f'Processing {len(ref_frames)} video frames...')
        
        h, w = im_ori_c1.shape[:2]
        im_ori_c1 = 255 - im_ori_c1
        
        simpleGraphs = merge_dic['simpleGraphs']
        edge_dic = merge_dic['EdgeDic']
        mask_dic = merge_dic['SplitMaskDic']
        G_merge_list = merge_dic['MergeList']
        
        m_list_ids = {} # Mask/ID list
        point_list_map = {} # Map ID to point list
        cnt = 0
        
        m_mask = np.zeros(im_ori_c1.shape, np.uint8)
        
        for k in G_merge_list:
             res = m_mask.copy()
             # Collect all points for this merged stroke
             merged_points = []
             
             for kk_cnt, kk in enumerate(k):
                 if kk_cnt == 0:
                     try:
                         padded_mask = mask_dic[kk][:, :, 0]
                     except:
                         padded_mask = mask_dic[kk]['update_mask'][:, :, 0]
                 else:
                     try:
                         padded_mask2 = mask_dic[kk][:, :, 0]
                     except:
                         padded_mask2 = mask_dic[kk]['update_mask'][:, :, 0]
                     padded_mask = cv2.bitwise_or(padded_mask, padded_mask2)
                 
                 res[padded_mask > MASK_TH] = 255
                 
                 # Collect points
                 if 'point_list' in mask_dic[kk]:
                     merged_points.extend(mask_dic[kk]['point_list'])
             
             m_list_ids[cnt] = res # We still need masks for matching
             point_list_map[cnt] = merged_points
             cnt += 1
        
        # Match strokes to frames (returns IDs now)
        msort = self.frame_matcher.match_strokes_to_frames(
            m_list_ids, ref_frames, (h, w)
        )
        
        sketches = self._generate_sketches(
             msort, point_list_map, len(ref_frames)
        )
        
        if len(sketches) > 0:
             print(f'✓ Generated {len(sketches)} sketches')
        
        return sketches
    
    def _point_to_vertex(self, p):
        """Convert [y, x] point to Vertex(x, y)."""
        # Note: input points are often [y, x] from numpy indices
        return Vertex(x=p[1], y=p[0], thickness=2.0, opacity=1.0, color=(0,0,0))

    def _create_linear_curve(self, p_start, p_end):
        """Create a linear Curve between two vertices."""
        # For linear curve, control points are same as start/end
        # or linearly interpolated. Sketchkit might expect specific handle placement.
        # Here we just place them at 1/3 and 2/3.
        p_c1 = Point(x=p_start.x + (p_end.x - p_start.x)/3.0, y=p_start.y + (p_end.y - p_start.y)/3.0)
        p_c2 = Point(x=p_start.x + 2*(p_end.x - p_start.x)/3.0, y=p_start.y + 2*(p_end.y - p_start.y)/3.0)
        return Curve(p_start, p_end, p_c1, p_c2)

    def _points_to_path(self, points):
        """Convert list of [y, x] points to a Path."""
        if len(points) < 2:
            return None
            
        curves = []
        # Downsample points specifically for vectorization to avoid huge SVG/paths
        # Simple skip, taking every Nth point
        step = max(1, len(points) // 100) # Ensure we don't hold too many points
        if step > 1:
            points = points[::step]
            # Ensure last point is included
            # if points[-1] != original_last: points.append(original_last) 
            
        for i in range(len(points) - 1):
             v_start = self._point_to_vertex(points[i])
             v_end = self._point_to_vertex(points[i+1])
             curves.append(self._create_linear_curve(v_start, v_end))
             
        return Path(curves)

    def _generate_sketches(self, msort, point_list_map, num_frames):
        """
        Generate Sketch objects for each frame.
        """
        sketches = []
        current_paths = []
        
        # msort keys are frame indices. 
        # We need to iterate through frames in order.
        mid_list = sorted(msort.keys())
        # Filter out special last_layid if not needed or treat it as final frame?
        # Usually last_layid=9999 collects unmatched strokes.
        # We should probably add them at the end or ignore. 
        # Let's assume user wants animation frames corresponding to video.
        
        # Map frame index to new strokes
        frame_new_strokes = {}
        for idx in mid_list:
            if idx == 9999: continue
            frame_new_strokes[idx] = msort[idx]

        # But we need to return 'num_frames' sketches? 
        # Or just sketches for frames where something happens?
        # The previous code returned `frames` list which I assume had length ~ num_frames?
        # Let's verify `_generate_colored_frames`:
        # it iterated `mid_list`. If `mid_list` only contains indices where strokes appear,
        # then we are skipping frames. 
        # Wait, `ref_frames` has all frames. `frame_matcher` iterates `ref_frames`.
        # `msort` keys are indices of `ref_frames`.
        
        # If we want to return a list of sketches corresponding to the animation flow:
        # We should probably iterate from 0 to max_frame_idx.
        
        max_frame = max([k for k in mid_list if k != 9999]) if len(mid_list) > 1 else 0
        # Or simple: just map what we have.
        
        # Re-reading `_generate_colored_frames`:
        # `for ii, idx_frame in enumerate(mid_list):`
        # It appends to `frames`. 
        # `frames.append(white_bg.copy())` 
        # It accumulates: `white_bg[mm] = color`.
        # So for each *key in msort* (which corresponds to a frame index), it produces ONE output frame?
        # No, `white_bg` is accumulated. 
        # It seems it produces a sequence of sketches, where each step adds new strokes.
        
        for idx_frame in mid_list:
             if idx_frame == 9999: 
                 # Unmatched strokes - add them at the very end or ignore?
                 # Let's add them to a final frame
                 stroke_ids = msort[idx_frame]
             else:
                 stroke_ids = msort[idx_frame]
            
             for sid in stroke_ids:
                 pts = point_list_map[sid]
                 path = self._points_to_path(pts)
                 if path:
                     current_paths.append(path)
             
             # Create sketch for this state
             # We need to deepcopy paths? Sketch holds references. 
             # Just creating new Sketch with list copy is enough structure-wise.
             # Note: logic requires accumulating paths.
             sketches.append(Sketch(paths=list(current_paths)))
             
        return sketches

