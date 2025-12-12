"""
Color generation utilities for creating visually distinct stroke colors.
"""

import numpy as np
import hsluv
import matplotlib.colors as mpl_colors


class ColorGenerator:
    """
    Generate perceptually distinct colors for stroke visualization.
    """
    
    def __init__(self, use_hsluv=True):
        """
        Initialize color generator.
        
        Args:
            use_hsluv: Whether to use HSLuv color space (more perceptually uniform)
        """
        self.use_hsluv = use_hsluv
        self.color_cache = None
    
    def get_color_map(self):
        """
        Get a list of distinct colors for visualization.
        
        Returns:
            list: List of RGB color tuples
        """
        if self.use_hsluv:
            return self._color_gene_hsluv()
        else:
            return self._gene_css4_cmap()
    
    def _color_gene_hsluv(self):
        """
        Generate colors using HSLuv color space for better perceptual uniformity.
        
        Returns:
            list: List of RGB color arrays
        """
        if self.color_cache is not None:
            return self.color_cache
        
        np.random.seed(666)
        sh = 20
        step = 5
        ss = round((100 - sh) / step)
        colors = []
        
        for s1 in range(0, 360, sh):
            for s2 in range(sh, 100, ss):
                s2 = min(s2, 100)
                s1 = min(s1, 360)
                c = [s1, s2, 70]
                v_rgb = hsluv.hsluv_to_rgb(c)
                rgb = np.round(np.array(v_rgb) * 255).astype(np.uint8)
                colors.append(rgb)
        
        # Shuffle for better distribution
        ind = np.random.choice(len(colors), len(colors), replace=False)
        colors2 = [colors[i] for i in ind]
        
        # Create light/dark pairs
        colors1 = [
            np.array(colors2[i // 2]) if i % 2 == 0 
            else 255 - np.array(colors2[i // 2])
            for i in range(2 * len(colors2))
        ]
        
        print('color_gene', len(colors1))
        self.color_cache = colors1
        return colors1
    
    def _gene_css4_cmap(self):
        """
        Generate colors from CSS4 color palette.
        
        Returns:
            list: List of RGB color tuples
        """
        cmaps = mpl_colors.CSS4_COLORS
        cmap = list(mpl_colors.CSS4_COLORS)
        
        # Filter out hard-to-see colors
        skip_color_lst = [
            'black', 'aliceblue', 'azure', 'dimgrey', 'ghostwhite',
            'honeydew', 'lightyellow', 'linen', 'mintcream', 'oldlace',
            'seashell', 'snow', 'white', 'whitesmoke'
        ]
        
        cmap2 = [mpl_colors.to_rgb(cmaps[i]) for i in cmap if i not in skip_color_lst]
        
        # Convert to BGR for OpenCV and filter out very light colors
        wlim = 220
        cmap3 = [
            (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
            for c in cmap2
            if int(c[2] * 255) < wlim and int(c[1] * 255) < wlim and int(c[0] * 255) < wlim
        ]
        
        return cmap3
