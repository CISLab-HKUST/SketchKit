"""
SENS Standalone Inference using TorchScript models.

This module provides SENS inference without requiring the original SENS codebase.
Only requires: torch, numpy, PIL, skimage

Models are automatically downloaded on first use, following the dataset download pattern.
"""

import os
import sys
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from skimage.measure import marching_cubes
except ImportError:
    try:
        from skimage.measure import marching_cubes_lewiner as marching_cubes
    except ImportError:
        marching_cubes = None

# Import download utilities from sketchkit
try:
    from sketchkit.utils.file import extract_files
except ImportError:
    extract_files = None

try:
    import gdown
except ImportError:
    gdown = None

# Google Drive folder ID for SENS TorchScript models
SENS_MODELS_GDRIVE_FOLDER_ID = "17pw-Op9bBlpcJuvocT4DtGpWAmnW_mU9"
REQUIRED_MODEL_FILES = [
    "sens_encoder.pt",
    "sens_occ_former_full.pt",
    "sens_from_gmm.pt",
    "sens_affine.pt",
    "sens_occ_head.pt",
    "sens_config.pt"
]


class SENSStandaloneInference:
    """
    Standalone SENS inference using exported TorchScript models.
    
    This provides the same functionality as the original SENS but without
    requiring the SENS source code.
    
    Required files in model_dir:
        - sens_encoder.pt
        - sens_occ_former_full.pt
        - sens_from_gmm.pt
        - sens_affine.pt
        - sens_occ_head.pt
        - sens_config.pt
    """
    
    def __init__(self, model_dir: Optional[str] = None, device: str = 'cpu', skip_integrity_check: bool = False):
        """
        Initialize SENS standalone inference.
        
        Models are automatically downloaded if not found, following the dataset download pattern.
        
        Args:
            model_dir: Directory containing exported TorchScript models.
                       If None, uses default cache location ~/.cache/sketchkit/sens_models/torchscript
            device: Computing device ('cpu' or 'cuda')
            skip_integrity_check: Skip integrity check and download (for testing)
        """
        self.device = torch.device(device)
        
        # Determine cache directory
        if model_dir is None:
            model_dir = str(Path.home() / ".cache" / "sketchkit" / "sens_models" / "torchscript")
        
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Check integrity and download if needed (following dataset pattern)
        if not skip_integrity_check:
            if not self._check_integrity():
                print(f"SENS TorchScript models not found or incomplete in {self.model_dir}.")
                print("Downloading from Google Drive...")
                self._download()
                if not self._check_integrity():
                    raise RuntimeError(
                        "Failed to download or verify SENS models. "
                        "Please check your internet connection or download manually from:\n"
                        f"https://drive.google.com/drive/folders/{SENS_MODELS_GDRIVE_FOLDER_ID}"
                    )
        
        self._load_models()
        self.sample_cache = {}
    
    def _check_integrity(self) -> bool:
        """Check if all required model files exist (following dataset pattern)."""
        for filename in REQUIRED_MODEL_FILES:
            file_path = self.model_dir / filename
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False
        return True
    
    def _download(self):
        """
        Download SENS models from Google Drive (following dataset download pattern).
        
        Downloads the entire folder and extracts it.
        """
        if gdown is None:
            raise RuntimeError(
                "gdown is required for automatic model download. "
                "Install with: pip install gdown\n"
                "Or download models manually from:\n"
                f"https://drive.google.com/drive/folders/{SENS_MODELS_GDRIVE_FOLDER_ID}"
            )
        
        print(f"Downloading SENS TorchScript models from Google Drive folder...")
        print(f"Folder ID: {SENS_MODELS_GDRIVE_FOLDER_ID}")
        
        try:
            # Use gdown to download the entire folder
            folder_url = f"https://drive.google.com/drive/folders/{SENS_MODELS_GDRIVE_FOLDER_ID}"
            print(f"Downloading from: {folder_url}")
            
            # gdown.download_folder downloads to a subdirectory with the folder name
            # Download to parent directory, then move files
            output_dir = self.model_dir.parent
            
            # Try download_folder (available in gdown 4.0+)
            if hasattr(gdown, 'download_folder'):
                gdown.download_folder(
                    id=SENS_MODELS_GDRIVE_FOLDER_ID,
                    output=str(output_dir),
                    quiet=False,
                    use_cookies=False
                )
                
                # Find downloaded files (they might be in a subdirectory)
                downloaded_files = list(output_dir.glob("**/*.pt"))
                
                if downloaded_files:
                    # Move files to model_dir
                    for file_path in downloaded_files:
                        target_path = self.model_dir / file_path.name
                        if file_path.parent != self.model_dir:
                            if target_path.exists():
                                file_path.unlink()  # Remove duplicate
                            else:
                                file_path.rename(target_path)
                    
                    # Clean up empty subdirectories
                    for subdir in output_dir.iterdir():
                        if subdir.is_dir() and subdir != self.model_dir:
                            try:
                                if not any(subdir.iterdir()):
                                    subdir.rmdir()
                            except:
                                pass
                    
                    print(f"✓ Models downloaded successfully to {self.model_dir}")
                else:
                    raise RuntimeError("Downloaded folder does not contain model files")
            else:
                # Fallback: manual download instructions
                raise RuntimeError(
                    "gdown version too old. Please upgrade: pip install --upgrade gdown>=4.0.0\n"
                    "Or download manually from the link above."
                )
                
        except Exception as e:
            # If folder download fails, provide manual download instructions
            print(f"\n⚠ Automatic download failed: {e}")
            print("\nPlease download manually:")
            print(f"1. Visit: https://drive.google.com/drive/folders/{SENS_MODELS_GDRIVE_FOLDER_ID}")
            print(f"2. Download all .pt files and place them in: {self.model_dir}")
            print(f"3. Required files: {', '.join(REQUIRED_MODEL_FILES)}")
            raise RuntimeError(
                f"Automatic download failed: {e}\n"
                f"Please download manually from: https://drive.google.com/drive/folders/{SENS_MODELS_GDRIVE_FOLDER_ID}"
            )
    
    def _find_model_dir(self, model_dir: Optional[str]) -> Optional[Path]:
        """Find the model directory (legacy method, kept for compatibility)."""
        if model_dir is not None:
            path = Path(model_dir)
            if path.exists() and (path / "sens_encoder.pt").exists():
                return path
        
        # Search default locations
        current_file = Path(__file__).resolve()
        search_paths = [
            current_file.parent.parent.parent.parent / "exported_sens_full",
            current_file.parent / "exported_sens_full",
            Path.home() / ".cache" / "sketchkit" / "sens_models" / "torchscript",
            Path("exported_sens_full"),
        ]
        
        for path in search_paths:
            if path.exists() and (path / "sens_encoder.pt").exists():
                return path
        
        return None
    
    def _load_models(self):
        """Load TorchScript models."""
        print(f"Loading SENS TorchScript models from: {self.model_dir}")
        
        # Load config
        config_path = self.model_dir / "sens_config.pt"
        if config_path.exists():
            self.config = torch.load(str(config_path), map_location=self.device, weights_only=True)
        else:
            self.config = {'dim_h': 512, 'num_gaussians': 16, 'as_tait_bryan': True}
        
        # Load models
        self.encoder = torch.jit.load(
            str(self.model_dir / "sens_encoder.pt"), 
            map_location=self.device
        )
        self.occ_former = torch.jit.load(
            str(self.model_dir / "sens_occ_former_full.pt"), 
            map_location=self.device
        )
        self.from_gmm = torch.jit.load(
            str(self.model_dir / "sens_from_gmm.pt"), 
            map_location=self.device
        )
        self.affine = torch.jit.load(
            str(self.model_dir / "sens_affine.pt"), 
            map_location=self.device
        )
        self.occ_head = torch.jit.load(
            str(self.model_dir / "sens_occ_head.pt"), 
            map_location=self.device
        )
        
        # Set eval mode
        for m in [self.encoder, self.occ_former, self.from_gmm, self.affine, self.occ_head]:
            m.eval()
        
        print("  ✓ All models loaded")
    
    def preprocess_sketch(self, image_input: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess sketch image for SENS.
        
        Args:
            image_input: Image path, numpy array, or PIL Image
            
        Returns:
            Preprocessed tensor [1, 1, 256, 256]
        """
        # Load image
        if isinstance(image_input, str):
            img = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
        else:
            img = image_input
        
        # Convert to RGB
        if img.mode != 'RGB':
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            else:
                img = img.convert('RGB')
        
        img_array = np.array(img, dtype=np.uint8)
        
        # Crop and center
        img_array = self._augment_cropped_square(img_array, res=256)
        
        # Convert to tensor
        sketch = torch.from_numpy(img_array).float()
        sketch = sketch.mean(-1) / 255.0  # Grayscale and normalize
        sketch = sketch.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        sketch = F.interpolate(sketch, 256, mode='bicubic', align_corners=True)
        sketch = sketch.gt(0.9).float()  # Binarize
        
        return sketch.to(self.device)
    
    def _augment_cropped_square(self, image: np.ndarray, res: int = 256) -> np.ndarray:
        """Crop to content and resize to square."""
        h, w = image.shape[:2]
        gray = image.mean(axis=2) if image.ndim == 3 else image
        
        non_white = gray < 250
        if not non_white.any():
            img_pil = Image.fromarray(image)
            return np.array(img_pil.resize((res, res), Image.BICUBIC))
        
        rows = np.any(non_white, axis=1)
        cols = np.any(non_white, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        rmin, rmax = max(0, rmin - 1), min(h - 1, rmax + 1)
        cmin, cmax = max(0, cmin - 1), min(w - 1, cmax + 1)
        
        cropped = image[rmin:rmax+1, cmin:cmax+1]
        ch, cw = cropped.shape[:2]
        side = max(ch, cw)
        
        if image.ndim == 3:
            square = np.full((side, side, 3), 255, dtype=np.uint8)
        else:
            square = np.full((side, side), 255, dtype=np.uint8)
        
        start_h, start_w = (side - ch) // 2, (side - cw) // 2
        square[start_h:start_h+ch, start_w:start_w+cw] = cropped
        
        return np.array(Image.fromarray(square).resize((res, res), Image.BICUBIC))
    
    @torch.no_grad()
    def encode_sketch(self, sketch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sketch to latent representation."""
        zh, out_cls = self.encoder(sketch)
        return zh, out_cls
    
    @torch.no_grad()
    def merge_zh_step_a(self, zh: torch.Tensor, mu: torch.Tensor, p: torch.Tensor,
                        phi: torch.Tensor, eigen: torch.Tensor) -> torch.Tensor:
        """
        Merge GMM information into zh (core SENS operation).
        """
        b = zh.shape[0]
        gp, g = mu.shape[1], mu.shape[2]
        
        # Apply softmax to phi
        phi_softmax = phi.softmax(2)
        
        # Reshape
        mu_flat = mu.view(b, gp * g, 3)
        eigen_flat = eigen.view(b, gp * g, 3)
        phi_flat = phi_softmax.view(b, gp * g, 1)
        p_flat = p.view(b, gp * g, 9)
        
        # Concatenate: mu(3) + p(9) + phi(1) + eigen(3) = 16
        z_gmm = torch.cat([mu_flat, p_flat, phi_flat, eigen_flat], dim=2)
        z_gmm = self.from_gmm(z_gmm)
        
        return zh + z_gmm
    
    @torch.no_grad()
    def prepare_features(self, zh: torch.Tensor, out_cls: torch.Tensor,
                         thresh_mask: float = 0.0) -> torch.Tensor:
        """Prepare features with full GMM merge."""
        # Get OccFormer output with GMM
        zh_out, mu, p, phi, eigen = self.occ_former(zh)
        
        # Merge GMM info
        zh_merged = self.merge_zh_step_a(zh_out, mu, p, phi, eigen)
        
        # Apply mask
        mask = out_cls.ge(thresh_mask)
        if mask.any():
            zh_masked = zh_merged[0][mask[0]].unsqueeze(0)
        else:
            zh_masked = zh_merged
        
        # Apply affine transformer
        return self.affine(zh_masked)
    
    def _get_grid_samples(self, res: int) -> torch.Tensor:
        """Get grid sample coordinates."""
        if res in self.sample_cache:
            samples = self.sample_cache[res].clone()
            samples[3, :] = 1.0
            return samples
        
        voxel_origin = torch.tensor([-1., -1., -1.])
        voxel_size = 2.0 / (res - 1)
        
        overall_index = torch.arange(0, res ** 3, dtype=torch.int64)
        samples = torch.ones(4, res ** 3)
        
        div_1 = torch.div(overall_index, res, rounding_mode='floor')
        samples[2, :] = (overall_index % res).float()
        samples[1, :] = (div_1 % res).float()
        samples[0, :] = (torch.div(div_1, res, rounding_mode='floor') % res).float()
        samples[:3] = samples[:3] * voxel_size + voxel_origin[:, None]
        
        samples = samples.to(self.device)
        self.sample_cache[res] = samples.clone()
        return samples
    
    @torch.no_grad()
    def compute_occupancy_grid(self, z: torch.Tensor, res: int = 128) -> torch.Tensor:
        """Compute occupancy grid."""
        samples = self._get_grid_samples(res)
        
        max_batch = 32 ** 3
        num_samples = samples.shape[1]
        
        for i in range((num_samples + max_batch - 1) // max_batch):
            start = i * max_batch
            end = min((i + 1) * max_batch, num_samples)
            coords = samples[:3, start:end].T.unsqueeze(0)
            occ = self.occ_head(coords, z)
            samples[3, start:end] = occ.squeeze()
        
        return samples[3].reshape(res, res, res)
    
    def mesh_from_occupancy(self, occ_values: torch.Tensor) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Generate mesh using marching cubes."""
        if marching_cubes is None:
            raise ImportError("scikit-image is required for mesh generation")
        
        res = occ_values.shape[0]
        voxel_size = 2.0 / (res - 1)
        voxel_origin = [-1., -1., -1.]
        
        occ_np = occ_values.cpu().numpy()
        
        try:
            verts, faces, _, _ = marching_cubes(occ_np, level=0.0, spacing=[voxel_size] * 3)
            verts[:, 0] += voxel_origin[0]
            verts[:, 1] += voxel_origin[1]
            verts[:, 2] += voxel_origin[2]
            return verts, faces
        except Exception as e:
            # Fallback to adaptive level
            return self._mesh_adaptive_level(occ_np, voxel_size, voxel_origin)
    
    def _mesh_adaptive_level(self, occ_np, voxel_size, voxel_origin):
        """Fallback with adaptive level."""
        min_val, max_val = occ_np.min(), occ_np.max()
        levels = [np.median(occ_np), (min_val + max_val) / 2,
                  np.percentile(occ_np, 25), np.percentile(occ_np, 10)]
        
        for level in levels:
            if min_val < level < max_val:
                try:
                    verts, faces, _, _ = marching_cubes(occ_np, level=level, spacing=[voxel_size] * 3)
                    if len(verts) > 100:
                        verts[:, 0] += voxel_origin[0]
                        verts[:, 1] += voxel_origin[1]
                        verts[:, 2] += voxel_origin[2]
                        return verts, faces
                except:
                    continue
        return None
    
    def infer(self, image_input: Union[str, np.ndarray, Image.Image], 
              res: int = 128) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Convert sketch to 3D mesh.
        
        Args:
            image_input: Image path, numpy array, or PIL Image
            res: Occupancy grid resolution (higher = more detail, slower)
            
        Returns:
            (vertices, faces) tuple or None if failed
        """
        # Preprocess
        sketch = self.preprocess_sketch(image_input)
        
        # Encode
        zh, out_cls = self.encode_sketch(sketch)
        
        # Prepare features with GMM merge
        z = self.prepare_features(zh, out_cls)
        
        # Compute occupancy and generate mesh
        occ = self.compute_occupancy_grid(z, res)
        
        return self.mesh_from_occupancy(occ)
    
    @staticmethod
    def save_mesh_obj(vertices: np.ndarray, faces: np.ndarray, output_path: str):
        """Save mesh as OBJ file."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def sketch_to_3d_sens_standalone(
    sketch_input: Union['Sketch', str, np.ndarray, Image.Image],
    output_path: Optional[str] = None,
    model_dir: Optional[str] = None,
    device: str = 'cpu',
    resolution: int = 128,
    **kwargs
) -> str:
    """
    Convert sketch to 3D model using SENS standalone inference.
    
    Args:
        sketch_input: Input sketch (Sketch object, file path, numpy array, or PIL Image)
        output_path: Output OBJ file path (auto-generated if None)
        model_dir: Directory containing TorchScript models
        device: Computing device ('cpu' or 'cuda')
        resolution: Occupancy grid resolution (default: 128)
        
    Returns:
        str: Path to output OBJ file
    """
    # Handle Sketch object input
    try:
        from sketchkit.core.sketch import Sketch
        if isinstance(sketch_input, Sketch):
            from sketchkit.renderer.cairo_renderer import CairoRenderer
            renderer = CairoRenderer(canvas_size=512, canvas_color=(1, 1, 1))
            raster_image = renderer.render(sketch_input, fit_canvas=True)
            if raster_image.dtype != np.uint8:
                raster_image = (raster_image * 255).astype(np.uint8)
            sketch_input = raster_image
    except ImportError:
        pass  # Sketch not available, treat as other types
    
    # Initialize inference
    inference = SENSStandaloneInference(model_dir=model_dir, device=device)
    
    # Run inference
    result = inference.infer(sketch_input, res=resolution)
    
    if result is None:
        raise RuntimeError("SENS inference failed to generate mesh")
    
    vertices, faces = result
    
    # Generate output path if not provided
    if output_path is None:
        import time
        cache_dir = Path.home() / ".cache" / "sketchkit" / "sens_output"
        cache_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        output_path = str(cache_dir / f"sens_{timestamp}.obj")
    
    # Save mesh
    inference.save_mesh_obj(vertices, faces, output_path)
    
    return output_path


# Check if standalone models are available
def is_standalone_available(model_dir: Optional[str] = None) -> bool:
    """Check if SENS standalone models are available."""
    try:
        if model_dir is None:
            model_dir = str(Path.home() / ".cache" / "sketchkit" / "sens_models" / "torchscript")
        model_path = Path(model_dir)
        # Check integrity
        for filename in REQUIRED_MODEL_FILES:
            if not (model_path / filename).exists():
                return False
        return True
    except:
        return False

