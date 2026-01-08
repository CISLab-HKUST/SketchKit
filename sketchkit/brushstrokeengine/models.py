"""
Model download utilities for Brushstroke Engine.
"""

import os
import gdown
from pathlib import Path
from typing import Optional
from sketchkit.utils.file import extract_files

# Model download URLs
# From README: https://nvidia-caat.s3.us-east-2.amazonaws.com/neube/models.zip
# Backup: https://drive.google.com/file/d/1OU7cEVeALBVaqDabj4uBQ1xJ8NrdOAI4/view?usp=sharing
MODELS_GOOGLE_DRIVE_ID = "1OU7cEVeALBVaqDabj4uBQ1xJ8NrdOAI4"
MODELS_S3_URL = "https://nvidia-caat.s3.us-east-2.amazonaws.com/neube/models.zip"


def get_models_dir() -> Path:
    """Get the default models directory path.
    
    Returns:
        Path to the models directory (sketchkit/brushstrokeengine/models/)
    """
    current_dir = Path(__file__).parent
    return current_dir / "models"


def download_models(
    models_dir: Optional[Path] = None,
    use_s3: bool = True,
    quiet: bool = False,
) -> Path:
    """
    Download and extract brushstroke engine models.
    
    Args:
        models_dir: Directory to download models to. If None, uses default location.
        use_s3: If True, use S3 URL; otherwise use Google Drive.
        quiet: If True, suppress download progress.
    
    Returns:
        Path to the models directory.
    """
    if models_dir is None:
        models_dir = get_models_dir()
    
    models_dir = Path(models_dir)
    neube_dir = models_dir / "neube"
    
    # Check if models already exist
    if neube_dir.exists() and (neube_dir / "style1").exists() and (neube_dir / "style2").exists():
        print(f"✅ Models already exist at {neube_dir}")
        return neube_dir
    
    # Create models directory
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download models.zip
    archive_path = models_dir / "models.zip"
    
    print(f"📥 Downloading brushstroke engine models...")
    print(f"   Destination: {models_dir}")
    
    try:
        if use_s3:
            # Try S3 first
            try:
                from sketchkit.utils.file import download_with_wget
                download_with_wget(
                    MODELS_S3_URL,
                    str(archive_path),
                    desc="Downloading models.zip",
                )
            except Exception as e:
                print(f"⚠️  S3 download failed, trying Google Drive: {e}")
                use_s3 = False
        
        if not use_s3:
            # Use Google Drive
            gdown.download(
                id=MODELS_GOOGLE_DRIVE_ID,
                output=str(archive_path),
                quiet=quiet,
            )
        
        print("📦 Extracting models...")
        extract_files(
            file_path=str(archive_path),
            output_dir=str(models_dir),
            remove_sourcefile=True,
        )

        # Some archives contain an extra top-level "models" directory:
        #   models.zip/
        #     models/
        #       neube/style1
        #       neube/style2
        # In that case, after extraction we get:
        #   models_dir/
        #     models/
        #       neube/...
        # Here we flatten that structure so the final layout is models_dir/neube/...
        nested_root = models_dir / "models"
        if nested_root.exists() and nested_root.is_dir():
            print(f"📂 Detected nested 'models' directory inside {models_dir}, flattening...")
            for item in nested_root.iterdir():
                target = models_dir / item.name
                if target.exists():
                    # 如果目标已存在，跳过避免覆盖（正常情况下不会发生）
                    continue
                item.rename(target)
            # 移除空的 nested_root 目录
            try:
                nested_root.rmdir()
            except OSError:
                # 如果目录非空（极少见），保留但不影响后续使用
                pass
        
        # Verify extraction
        if not (neube_dir / "style1").exists() or not (neube_dir / "style2").exists():
            raise RuntimeError(
                f"Model extraction incomplete. Expected directories not found in {neube_dir}"
            )
        
        print(f"✅ Models downloaded and extracted successfully!")
        print(f"   Location: {neube_dir}")
        return neube_dir
        
    except Exception as e:
        print(f"❌ Failed to download models: {e}")
        if archive_path.exists():
            archive_path.unlink()
        raise


def check_models_available(model_name: str = "style2", models_dir: Optional[Path] = None) -> bool:
    """
    Check if models are available for the specified model name.
    
    Args:
        model_name: Name of the model ('style1' or 'style2').
        models_dir: Directory containing models. If None, uses default location.
    
    Returns:
        True if models are available, False otherwise.
    """
    if models_dir is None:
        models_dir = get_models_dir()
    
    model_path = Path(models_dir) / "neube" / model_name
    checkpoint_path = model_path / "snapshot.pkl"
    
    return checkpoint_path.exists()


def get_model_path(model_name: str = "style2", models_dir: Optional[Path] = None) -> Path:
    """
    Get the path to a model checkpoint.
    
    Args:
        model_name: Name of the model ('style1' or 'style2').
        models_dir: Directory containing models. If None, uses default location.
    
    Returns:
        Path to the model checkpoint directory.
    
    Raises:
        FileNotFoundError: If the model is not found.
    """
    if models_dir is None:
        models_dir = get_models_dir()
    
    model_path = Path(models_dir) / "neube" / model_name
    checkpoint_path = model_path / "snapshot.pkl"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {model_path}. "
            f"Please download models first using download_models()."
        )
    
    return model_path

