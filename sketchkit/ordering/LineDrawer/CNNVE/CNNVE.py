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
from sketchkit.vectorization.LineDrawer.CNNVE.utils.image_utils import criterion_mask_overlap

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
    
    def run(self, img: np.ndarray, video_path: str) -> list[np.ndarray]:
        """
        Generate animation frames from image and video.
        
        Args:
            img (np.ndarray): Input image as a 2D numpy array (grayscale).
            video_path (str): Path to video file for frame synchronization.
            
        Returns:
            list[np.ndarray]: List of generated frames
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
            list: List of generated frames
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
        
        m_mask = np.zeros(im_ori_c1.shape, np.uint8)
        m_list = {}
        cnt = 0
        
        for k in G_merge_list:
            res = m_mask.copy()
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
            
            m_list[cnt] = res
            cnt += 1
        
        msort = self.frame_matcher.match_strokes_to_frames(
            m_list, ref_frames, (h, w)
        )
        
        frames = self._generate_colored_frames(
            msort, im_ori_c1, ref_frames, (h, w)
        )
        
        if len(frames) > 0:
             print(f'✓ Generated {len(frames)} frames')
        
        return frames
    
    def _generate_colored_frames(self, msort, im_ori_c1, ref_frames, size):
        """
        Generate colored animation frames.
        
        Args:
            msort: Sorted strokes by frame
            im_ori_c1: Original image
            ref_frames: Reference video frames
            size: Image size (h, w)
            
        Returns:
            list: List of colored frames
        """
        h, w = size
        frames = []
        
        no_bg = np.zeros((h, w, 3), np.uint8)
        white_bg = 255 - no_bg
        
        cmaps = self.color_generator.get_color_map()
        cnt = -1
        
        mid_list = sorted(msort.keys())
        
        for ii, idx_frame in enumerate(mid_list):
            stroke_m_list = msort[idx_frame]
            
            for stroke_m in stroke_m_list:
                cnt += 1
                color = cmaps[cnt % len(cmaps)]
                mm = stroke_m > 50
                
                white_bg[mm] = color
                frames.append(white_bg.copy())
        
        return frames

