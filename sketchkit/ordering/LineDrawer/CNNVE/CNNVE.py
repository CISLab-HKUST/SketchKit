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
    
    def run(self, img: np.ndarray, video_path: str) -> Sketch:
        """
        Generate a single Sketch with ordered paths from image and video.
        
        Args:
            img (np.ndarray): Input image as a 2D numpy array (grayscale).
            video_path (str): Path to video file for frame synchronization.
            
        Returns:
            Sketch: Generated Sketch object with paths ordered by drawing sequence.
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
            Sketch: Generated Sketch object
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
                     merged_points.append(mask_dic[kk]['point_list'])
             
             m_list_ids[cnt] = res # We still need masks for matching
             point_list_map[cnt] = merged_points
             cnt += 1
        
        # Match strokes to frames (returns IDs now)
        msort = self.frame_matcher.match_strokes_to_frames(
            m_list_ids, ref_frames, (h, w)
        )
        
        sketch = self._generate_ordered_sketch(
             msort, point_list_map
        )
        
        print(f'✓ Generated ordered sketch with {sketch.path_num} paths')
        
        return sketch
    
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

    def _generate_ordered_sketch(self, msort, point_list_map):
        """
        Generate a single Sketch object with paths ordered by frame appearance.
        """
        ordered_paths = []
        
        mid_list = sorted(msort.keys())
        
        for idx_frame in mid_list:
             stroke_ids = msort[idx_frame]
            
             for sid in stroke_ids:
                 pts_list = point_list_map[sid]
                 for pts in pts_list:
                     path = self._points_to_path(pts)
                     if path:
                         ordered_paths.append(path)
             
        return Sketch(paths=ordered_paths)

