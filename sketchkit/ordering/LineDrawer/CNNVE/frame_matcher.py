"""
Frame matching utilities for synchronizing strokes with video frames.
"""

import cv2
import numpy as np
from sketchkit.vectorization.LineDrawer.CNNVE.utils.image_utils import criterion_mask_overlap


class FrameMatcher:
    """
    Match extracted strokes to video frames based on overlap.
    """
    
    def __init__(self):
        pass
    
    def match_strokes_to_frames(self, m_list, ref_frames, size):
        """
        Match each stroke to the video frame where it first appears.
        
        Args:
            m_list: Dictionary of stroke masks
            ref_frames: List of reference video frames
            size: Image size tuple (h, w)
            
        Returns:
            dict: Dictionary mapping frame indices to lists of strokes
        """
        h, w = size
        msort = {}
        last_layid = 9999
        msort[last_layid] = []
        
        print('frame match start', len(m_list.keys()), 'ref_frames', len(ref_frames))
        
        for id in m_list.keys():
            stroke_m = m_list[id]
            b_matched = False
            
            for idx_frame, im_frame in enumerate(ref_frames):
                im_frame = 255 - cv2.resize(im_frame, (w, h), interpolation=cv2.INTER_AREA)
                stroke_m_small = cv2.resize(stroke_m, (w, h), interpolation=cv2.INTER_AREA)
                
                check_res = criterion_mask_overlap(stroke_m_small, im_frame, 0.3, 0, 0)
                
                if check_res[0]:
                    if check_res[1] < 0.3:
                        continue
                else:
                    continue
                
                if idx_frame not in msort.keys():
                    msort[idx_frame] = []
                
                # Store the ID instead of the mask
                msort[idx_frame].append(id)
                b_matched = True
                break
            
            if not b_matched:
                msort[last_layid].append(id)
        
        return msort
