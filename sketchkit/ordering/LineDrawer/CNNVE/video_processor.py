"""
Video processing utilities for extracting and filtering frames from MP4 files.
"""

import os
import glob
import cv2
import numpy as np
from PIL import Image, ImageOps
from sketchkit.vectorization.LineDrawer.CNNVE.utils.image_utils import is_background_black_undopainst


def is_increasing_relation(imgA, imgB):
    """
    Checks if imgB is an \"increasing\" version of imgA.
    
    Args:
        imgA: Previous image
        imgB: Next image
        
    Returns:
        bool: True if B includes most of A
    """
    _, imgA = cv2.threshold(255 - np.array(imgA, np.uint8), 127, 255, cv2.THRESH_BINARY)
    _, imgB = cv2.threshold(255 - np.array(imgB, np.uint8), 127, 255, cv2.THRESH_BINARY)
    
    A_pixels = np.count_nonzero(imgA)
    union = cv2.bitwise_and(imgA, imgB)
    union_pixels = np.count_nonzero(union)
    
    condition1 = union_pixels / (A_pixels + 0.01) > 0.6
    return condition1


def add_pil_border(original_image, border_width=3, border_color=(255, 0, 0)):
    """
    Adds a colored border to a PIL Image.
    
    Args:
        original_image: PIL Image
        border_width: Border width in pixels
        border_color: RGB color tuple
        
    Returns:
        PIL Image with border
    """
    bordered_image = ImageOps.expand(
        original_image.convert('RGB'),
        border=border_width,
        fill=border_color
    )
    return bordered_image


def get_match(images, indir):
    """
    Filters frames to find key frames in drawing process.
    
    Args:
        images: List of image frames
        indir: Output directory
        
    Returns:
        list: Filtered key frames
    """
    images2 = []
    images2_idx = []
    
    for i_img in range(len(images)):
        A = images[i_img]
        if i_img >= len(images) - 1:
            images2.append(A.copy())
            images2_idx.append(i_img)
            continue
        
        B = images[-1]
        C = images[i_img + 1]
        b1 = is_increasing_relation(A, B)
        b2 = is_increasing_relation(A, C)
        
        if b1 and b2:
            images2_idx.append(i_img)
            images2.append(A.copy())
    
    # Save visualization
    images3 = []
    for id, img in enumerate(images):
        if id in images2_idx:
            imgbb = add_pil_border(Image.fromarray(img), border_width=3, border_color=(0, 255, 0))
        else:
            imgbb = add_pil_border(Image.fromarray(img))
        images3.append(imgbb.copy())
    
    if len(images3) > 0:
        widths, heights = zip(*(img.size for img in images3))
        total_width = sum(widths)
        max_height = max(heights)
        
        merged_img = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images3:
            merged_img.paste(img, (x_offset, 0))
            x_offset += img.size[0]
            img.close()
        
        output_path = os.path.join(indir, "remove-imglist.png")
        merged_img.save(output_path)
    
    return images2


class VideoProcessor:
    """
    Process video files to extract and filter frames.
    """
    
    def __init__(self):
        pass
    
    def mp4_to_imgs(self, src_mp4, b_gt, ref):
        """
        Extract frames from MP4 file.
        
        Args:
            src_mp4: Path to MP4 file
            b_gt: Whether to use ground truth
            ref: Reference image
            
        Returns:
            list: Filtered frames
        """
        #print(src_mp4, 'src_mp4,b_gt=', b_gt)
        
        if b_gt:
            # Ground truth not implemented for now
            return []
        
        import tempfile
        outdir = tempfile.mkdtemp(prefix="video_frames_")
        
        # Use OpenCV instead of ffmpeg for better compatibility
        cap = cv2.VideoCapture(src_mp4)
        if not cap.isOpened():
            print(f"Error: Could not open video file {src_mp4}")
            return []
            
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = os.path.join(outdir, f"image{frame_idx + 1:02d}.png")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
            
        cap.release()
        
        return self.imgs_to_mask(outdir, ref)
    
    def imgs_to_mask(self, indir, linedrawing=None):
        """
        Load and filter image frames from directory.
        
        Args:
            indir: Directory with frames
            linedrawing: Reference final image
            
        Returns:
            list: Filtered frames
        """
        m_th = 128
        logdir = indir + '/out'
        os.makedirs(logdir, exist_ok=True)
        
        fs = glob.glob(indir + '/*.png')
        imglst = []
        imglst_fname = []
        
        for f in fs:
            if 'image' in f:
                im = cv2.imread(f, 0)
                im[im > m_th] = 255
                im[im <= m_th] = 0
                if is_background_black_undopainst(255 - im):
                    imglst.append(im)
                    imglst_fname.append(f)
        
        if linedrawing is None:
            ref = 255 - imglst[-1]
        else:
            h, w = imglst[-1].shape[:2]
            ref = cv2.resize(linedrawing, (w, h), interpolation=cv2.INTER_AREA)
        
        print('imgs_to_mask2', imglst_fname)
        print('linedrawing', ref.shape, imglst[-1].shape)
        
        # Take last 15 frames
        imglst = imglst[-15:]
        
        match_lst2 = get_match(imglst, indir)
        return match_lst2
