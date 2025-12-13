import os
import sys
from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as nnf
from threading import Lock
import tempfile


def _download_file(url: str, dest_path: Path, chunk_size: int = 8192):
    try:
        import requests
    except ImportError:
        raise ImportError("requests library required: pip install requests")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rDownloading: {percent:.1f}%", end='', flush=True)
    print()


def _download_models(model_dir: Path):
    SKETCH_MODEL_ID = "15b98DQL3ebL_SaccRys0zJimCxppxaVo"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = model_dir / "SENS_weights.zip"
    if not zip_path.exists():
        try:
            import gdown
            gdown.download(id=SKETCH_MODEL_ID, output=str(zip_path), quiet=False)
        except ImportError:
            url = f"https://drive.google.com/uc?export=download&id={SKETCH_MODEL_ID}"
            _download_file(url, zip_path)
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}")
    
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)
    
    old_sketch_dir = model_dir / "sketch2spaghetti_chairs"
    old_spaghetti_dir = model_dir / "occ_gmm_chairs_sym_hard"
    
    if not old_sketch_dir.exists() or not old_spaghetti_dir.exists():
        raise FileNotFoundError("Invalid model structure after extraction")
    
    spaghetti_checkpoint_dir = model_dir / "spaghetti" / "checkpoints" / "occ_gmm_chairs_sym_hard"
    if not spaghetti_checkpoint_dir.exists():
        spaghetti_checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.move(str(old_spaghetti_dir), str(spaghetti_checkpoint_dir))
    
    sketch_model_path = model_dir / "sketch2spaghetti.pt"
    if not sketch_model_path.exists():
        try:
            _export_torchscript_model(model_dir, old_sketch_dir)
        except Exception as e:
            if not sketch_model_path.exists():
                raise RuntimeError(
                    f"TorchScript model not found and auto-export failed.\n"
                    f"Error: {e}\n"
                    f"Please manually place sketch2spaghetti.pt at: {sketch_model_path}\n"
                    f"Or install missing dependencies (e.g., pip install igl) and retry."
                )


def _export_torchscript_model(model_dir: Path, sketch_checkpoint_dir: Path):
    current_file = Path(__file__)
    spaghetti_deps_path = current_file.parent / 'sens' / 'spaghetti_deps'
    if spaghetti_deps_path.exists():
        sys.path.insert(0, str(spaghetti_deps_path))
    
    try:
        from options import SketchOptions
        from utils import train_utils
        from models.sketch_model import Sketch2Spaghetti
    except ImportError:
        raise ImportError("Cannot import SENS modules for TorchScript export")
    
    class Sketch2SpaghettiWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x: torch.Tensor) -> tuple:
            zh, cls, z_mid = self.model(x, return_mid=True)
            return zh, cls, z_mid
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    opt = SketchOptions(tag="chairs", spaghetti_tag="chairs_sym_hard")
    opt.device = device
    opt.cp_folder = str(sketch_checkpoint_dir.parent)
    
    model, opt = train_utils.model_lc(opt)
    model.eval()
    
    wrapped_model = Sketch2SpaghettiWrapper(model)
    wrapped_model.eval()
    dummy_input = torch.zeros(1, 1, 256, 256).to(device)
    
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapped_model, dummy_input, strict=False)
    
    output_path = model_dir / "sketch2spaghetti.pt"
    traced_model.save(str(output_path))


def _ensure_models(model_cache_dir: Optional[str] = None) -> Tuple[Path, Path]:
    if model_cache_dir is None:
        model_cache_dir = os.environ.get('SENS_MODEL_DIR', None)
        if model_cache_dir is None:
            current_file = Path(__file__)
            possible_paths = [
                current_file.parent.parent.parent.parent / 'models' / 'sens',
                Path.home() / '.sketchkit' / 'models' / 'sens',
                Path(tempfile.gettempdir()) / 'sketchkit_models' / 'sens',
            ]
            for path in possible_paths:
                if path.exists():
                    model_cache_dir = str(path)
                    break
            if model_cache_dir is None:
                model_cache_dir = str(possible_paths[0])
    
    model_dir = Path(model_cache_dir)
    sketch_model_path = model_dir / "sketch2spaghetti.pt"
    spaghetti_checkpoint_dir = model_dir / "spaghetti" / "checkpoints" / "occ_gmm_chairs_sym_hard"
    
    if not sketch_model_path.exists() or not spaghetti_checkpoint_dir.exists():
        try:
            _download_models(model_dir)
            sketch_model_path = model_dir / "sketch2spaghetti.pt"
            spaghetti_checkpoint_dir = model_dir / "spaghetti" / "checkpoints" / "occ_gmm_chairs_sym_hard"
        except Exception as e:
            if "TorchScript" in str(e) and not sketch_model_path.exists():
                raise RuntimeError(
                    f"Model download completed but TorchScript export failed.\n"
                    f"Please manually place sketch2spaghetti.pt at: {sketch_model_path}\n"
                    f"Original error: {e}"
                )
            raise
    
    if not sketch_model_path.exists():
        raise FileNotFoundError(f"Sketch2Spaghetti model not found: {sketch_model_path}")
    if not spaghetti_checkpoint_dir.exists():
        raise FileNotFoundError(f"SPAGHETTI checkpoint not found: {spaghetti_checkpoint_dir}")
    
    return sketch_model_path, spaghetti_checkpoint_dir


_MODEL_CACHE = {}
_CACHE_LOCK = Lock()


class MinimalSketch2Mesh:
    def __init__(
        self,
        sketch_model_path: str,
        spaghetti_checkpoint_dir: str,
        device: Optional[torch.device] = None,
        lazy_load: bool = True,
        use_cache: bool = True
    ):
        self.sketch_model_path = Path(sketch_model_path).resolve()
        self.spaghetti_checkpoint_dir = Path(spaghetti_checkpoint_dir).resolve()
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_cache = use_cache
        self._sketch_model = None
        self._spaghetti_inference = None
        
        if not lazy_load:
            self._load_models()
    
    @classmethod
    def get_cached(cls, sketch_model_path: str, spaghetti_checkpoint_dir: str, device: Optional[torch.device] = None):
        device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        sketch_path = Path(sketch_model_path).resolve()
        spaghetti_dir = Path(spaghetti_checkpoint_dir).resolve()
        cache_key = (str(sketch_path), str(spaghetti_dir), str(device))
        
        with _CACHE_LOCK:
            if cache_key not in _MODEL_CACHE:
                _MODEL_CACHE[cache_key] = cls(sketch_model_path, spaghetti_checkpoint_dir, device, lazy_load=False, use_cache=True)
            return _MODEL_CACHE[cache_key]
    
    def _load_sketch_model(self):
        if self._sketch_model is None:
            if self.use_cache:
                cache_key = ('sketch', str(self.sketch_model_path), str(self.device))
                try:
                    if _CACHE_LOCK.acquire(blocking=False):
                        try:
                            if cache_key in _MODEL_CACHE:
                                self._sketch_model = _MODEL_CACHE[cache_key]
                                return
                        finally:
                            _CACHE_LOCK.release()
                except Exception:
                    pass
            
            if not self.sketch_model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.sketch_model_path}")
            
            self._sketch_model = torch.jit.load(str(self.sketch_model_path), map_location=self.device)
            self._sketch_model.eval()
            
            if self.use_cache:
                cache_key = ('sketch', str(self.sketch_model_path), str(self.device))
                if _CACHE_LOCK.acquire(blocking=False):
                    try:
                        _MODEL_CACHE[cache_key] = self._sketch_model
                    finally:
                        _CACHE_LOCK.release()
    
    def _load_spaghetti_model(self):
        if self._spaghetti_inference is None:
            current_file = Path(__file__)
            spaghetti_deps_path = current_file.parent / 'sens' / 'spaghetti_deps'
            
            if spaghetti_deps_path.exists() and str(spaghetti_deps_path) not in sys.path:
                sys.path.insert(0, str(spaghetti_deps_path))
            
            try:
                from options import Options
                from ui_sketch import occ_inference
                import constants
                
                checkpoint_dir_str = str(self.spaghetti_checkpoint_dir)
                
                original_checkpoints_root = constants.CHECKPOINTS_ROOT
                checkpoint_parent = str(self.spaghetti_checkpoint_dir.parent)
                if not checkpoint_parent.endswith('/'):
                    checkpoint_parent += '/'
                constants.CHECKPOINTS_ROOT = checkpoint_parent
                
                opt = Options(tag="chairs_sym_hard")
                opt.device = self.device
                opt = opt.load()
                
                if opt is None:
                    raise FileNotFoundError(f"Cannot load SPAGHETTI config: {self.spaghetti_checkpoint_dir}")
                
                if not Path(opt.cp_folder).exists():
                    opt.cp_folder = checkpoint_dir_str
                
                self._spaghetti_inference = occ_inference.Inference(opt)
                constants.CHECKPOINTS_ROOT = original_checkpoints_root
            except ImportError as e:
                raise ImportError(f"Cannot import SPAGHETTI modules: {e}")
    
    def _load_models(self):
        self._load_sketch_model()
        self._load_spaghetti_model()
    
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        elif image.dtype != np.uint8:
            raise ValueError("Input image must be uint8")
        
        h, w = image.shape[:2]
        non_white = np.any(image < 250, axis=2)
        
        if not np.any(non_white):
            side = max(h, w)
            result = np.full((side, side, 3), 255, dtype=np.uint8)
            if h < w:
                start = (side - h) // 2
                result[start:start+h, :] = image
            else:
                start = (side - w) // 2
                result[:, start:start+w] = image
            image = result
        else:
            rows = np.any(non_white, axis=1)
            cols = np.any(non_white, axis=0)
            top, bottom = np.where(rows)[0][[0, -1]]
            left, right = np.where(cols)[0][[0, -1]]
            
            top, bottom = max(0, top-1), min(h-1, bottom+1)
            left, right = max(0, left-1), min(w-1, right+1)
            image = image[top:bottom+1, left:right+1]
            
            h, w = image.shape[:2]
            side = max(h, w)
            result = np.full((side, side, 3), 255, dtype=np.uint8)
            if h < w:
                start = (side - h) // 2
                result[start:start+h, :] = image
            else:
                start = (side - w) // 2
                result[:, start:start+w] = image
            image = result
        
        image = Image.fromarray(image)
        image = image.resize((256, 256), Image.Resampling.BICUBIC)
        return np.array(image)
    
    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image).float()
        tensor = tensor.mean(-1) / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = nnf.interpolate(tensor, 256, mode='bicubic', align_corners=True)
        tensor = (tensor > 0.9).float()
        return tensor.to(self.device)
    
    def infer(self, image: Union[str, np.ndarray, Image.Image], resolution: int = 128, threshold: float = 0.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._sketch_model is None or self._spaghetti_inference is None:
            self._load_models()
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        processed = self._preprocess_image(image)
        sketch_tensor = self._image_to_tensor(processed)
        
        with torch.no_grad():
            zh, cls, z_mid = self._sketch_model(sketch_tensor)
            cls = cls.squeeze().sigmoid()
        
        with torch.no_grad():
            out_z, gmms = self._spaghetti_inference.model.occ_former.forward_mid([zh])
            out_z = self._spaghetti_inference.model.merge_zh_step_a(out_z, gmms)
            mask = cls.ge(threshold)
            out_z, _ = self._spaghetti_inference.model.affine_transformer.forward_with_attention(out_z[0][mask].unsqueeze(0))
            mesh = self._spaghetti_inference.get_mesh(out_z[0], resolution, None)
        
        if mesh is None:
            return None
        
        vertices = mesh[0].detach().cpu().numpy()
        faces = mesh[1].detach().cpu().numpy() if mesh[1] is not None else np.array([], dtype=np.int64).reshape(0, 3)
        return vertices, faces


_model_instance_cache = {}


def _get_model_instance(model_cache_dir: Optional[str] = None, device: str = 'cpu'):
    sketch_model_path, spaghetti_checkpoint_dir = _ensure_models(model_cache_dir)
    cache_key = (str(sketch_model_path), str(spaghetti_checkpoint_dir), device)
    
    if cache_key not in _model_instance_cache:
        _model_instance_cache[cache_key] = MinimalSketch2Mesh.get_cached(
            sketch_model_path=str(sketch_model_path),
            spaghetti_checkpoint_dir=str(spaghetti_checkpoint_dir),
            device=torch.device(device)
        )
    
    return _model_instance_cache[cache_key]


def sens_inference_minimal(
    sketch_input: Union[str, np.ndarray, Image.Image],
    output_path: str,
    model_cache_dir: Optional[str] = None,
    device: str = 'cpu',
    resolution: int = 128,
    threshold: float = 0.0
) -> str:
    model = _get_model_instance(model_cache_dir, device)
    
    if isinstance(sketch_input, str):
        image = Image.open(sketch_input).convert('RGB')
    elif isinstance(sketch_input, np.ndarray):
        if sketch_input.dtype != np.uint8:
            sketch_input = (sketch_input * 255).astype(np.uint8)
        if len(sketch_input.shape) == 2:
            sketch_input = np.stack([sketch_input] * 3, axis=-1)
        image = Image.fromarray(sketch_input)
    elif isinstance(sketch_input, Image.Image):
        image = sketch_input.convert('RGB')
    else:
        raise TypeError(f"Unsupported input type: {type(sketch_input)}")
    
    result = model.infer(image, resolution=resolution, threshold=threshold)
    
    if result is None:
        raise RuntimeError("Inference failed: no valid mesh generated")
    
    vertices, faces = result
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if faces is not None and len(faces) > 0:
            for face in faces:
                if len(face) == 3:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                else:
                    face_str = ' '.join([str(int(f)+1) for f in face])
                    f.write(f"f {face_str}\n")
    
    return output_path


def sketch_to_3d_sens_full(
    sketch_input: Union['Sketch', str, np.ndarray],
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    import tempfile
    import time
    
    if hasattr(sketch_input, 'render'):
        try:
            from sketchkit.renderer.cairo_renderer import CairoRenderer
            renderer = CairoRenderer(canvas_size=512, canvas_color=(1, 1, 1))
            raster_image = renderer.render(sketch_input, fit_canvas=True)
            if raster_image.ndim == 3:
                raster_image = np.dot(raster_image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
            sketch_input = raster_image
        except ImportError:
            raise ImportError("sketchkit.renderer required to render Sketch objects")
    
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time() * 1000)
        output_path = os.path.join(temp_dir, f"sens_output_{timestamp}.obj")
    
    model_cache_dir = kwargs.pop('model_cache_dir', None)
    device = kwargs.pop('device', 'cpu')
    resolution = kwargs.pop('resolution', 128)
    threshold = kwargs.pop('threshold', 0.0)
    
    return sens_inference_minimal(
        sketch_input=sketch_input,
        output_path=output_path,
        model_cache_dir=model_cache_dir,
        device=device,
        resolution=resolution,
        threshold=threshold
    )


__all__ = ['sens_inference_minimal', 'sketch_to_3d_sens_full']
