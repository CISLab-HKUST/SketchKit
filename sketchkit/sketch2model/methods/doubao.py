"""
Doubao-Seed3D API implementation for sketch-to-3D model conversion.
"""

import numpy as np
from PIL import Image
import os
import base64
import io
import time
import requests
import zipfile
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


def sketch_to_3d_doubao_impl(
    raster_sketch: np.ndarray,
    output_path: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convert sketch to 3D model using Doubao-Seed3D API.
    
    Args:
        raster_sketch: Rasterized sketch image
        output_path: Output file path (if None, will be auto-generated)
        api_key: Doubao API key (if None, reads from DOUBAO_API_KEY env var or ~/.sketchkit/config)
        api_base: API base URL (if None, uses default)
        **kwargs: Additional parameters
            - model_name: Model name (default: "doubao-seed3d-1-0-250928")
            - file_format: Output format (default: "glb")
            - subdivision_level: Subdivision level (default: "medium")
            - max_wait_time: Maximum wait time in seconds (default: 600)
            - poll_interval: Polling interval in seconds (default: 5)
    
    Returns:
        Generated 3D model file path
    """
    try:
        if api_key is None:
            api_key = os.environ.get('DOUBAO_API_KEY')
            
            if api_key is None:
                home_config_path = Path.home() / '.sketchkit' / 'config'
                if home_config_path.exists() and HAS_DOTENV:
                    dotenv.load_dotenv(home_config_path, override=False)
                    api_key = os.environ.get('DOUBAO_API_KEY')
            
            if api_key is None:
                raise ValueError(
                    "DOUBAO_API_KEY not found. Please set it via parameter, "
                    "environment variable DOUBAO_API_KEY, or ~/.sketchkit/config file"
                )
        
        model_name = kwargs.get('model_name', 'doubao-seed3d-1-0-250928')
        subdivision_level = kwargs.get('subdivision_level', 'medium')
        file_format = kwargs.get('file_format', 'glb')
        max_wait_time = kwargs.get('max_wait_time', 600)
        poll_interval = kwargs.get('poll_interval', 5)
        
        # API endpoint
        if api_base is None:
            api_base = "https://ark.cn-beijing.volces.com/api/v3/contents/generations"
        create_task_url = f"{api_base}/tasks"
        
        # Convert image to base64
        if len(raster_sketch.shape) == 3:
            image = Image.fromarray(raster_sketch)
        else:
            image = Image.fromarray(raster_sketch, mode='L').convert('RGB')
        
        # Save to memory as PNG
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/png;base64,{image_base64}"
        
        # Build request payload
        # Note: doubao-seed3d-1-0 model does not support text prompt, only image
        payload = {
            "model": model_name,
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
        
        # Try to add parameters to payload (if API supports)
        if subdivision_level:
            payload["subdivision_level"] = subdivision_level
        if file_format:
            payload["file_format"] = file_format
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Create task
        print(f"[Doubao Seed3D] Creating task...")
        print(f"[Doubao Seed3D] Model: {model_name}, Format: {file_format}, Subdivision: {subdivision_level}")
        
        response = requests.post(create_task_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        task_data = response.json()
        task_id = task_data.get('id')
        
        if not task_id:
            raise ValueError(f"Failed to create task: No task ID found in response. Response: {task_data}")
        
        print(f"[Doubao Seed3D] Task created, ID: {task_id}")
        
        # Poll task status
        query_url = f"{api_base}/tasks/{task_id}"
        start_time = time.time()
        
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > max_wait_time:
                raise TimeoutError(f"Task timeout: Wait time exceeded {max_wait_time} seconds")
            
            print(f"[Doubao Seed3D] Querying task status... (waited {int(elapsed_time)} seconds)")
            
            response = requests.get(query_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            task_info = response.json()
            status = task_info.get('status')
            
            # Handle completion status (including 'completed' and 'succeeded')
            if status == 'completed' or status == 'succeeded':
                print(f"[Doubao Seed3D] Task completed! Status: {status}")
                
                # Get result file URL
                result_url = task_info.get('result_url') or task_info.get('output_url')
                if not result_url:
                    # Try to get from content field (Doubao API response format)
                    content = task_info.get('content')
                    if isinstance(content, dict):
                        result_url = content.get('file_url') or content.get('url') or content.get('result_url')
                
                if not result_url:
                    # Try to get from other fields
                    result = task_info.get('result') or task_info.get('output')
                    if isinstance(result, dict):
                        result_url = result.get('url') or result.get('file_url')
                    elif isinstance(result, str):
                        # If result is a string, it might be a URL
                        result_url = result
                
                # If still not found, try to get from contents field (plural form)
                if not result_url:
                    contents = task_info.get('contents', [])
                    if contents and isinstance(contents, list):
                        for content in contents:
                            if isinstance(content, dict):
                                result_url = content.get('url') or content.get('file_url') or content.get('result_url')
                                if result_url:
                                    break
                
                if not result_url:
                    # Print full response for debugging
                    print(f"[Doubao Seed3D] Debug info - Full response: {task_info}")
                    raise ValueError(f"Task completed but result URL not found. Response: {task_info}")
                
                # Download result file
                print(f"[Doubao Seed3D] Downloading result file: {result_url}")
                file_response = requests.get(result_url, timeout=60)
                file_response.raise_for_status()
                
                # Save file
                if output_path is None:
                    temp_dir = tempfile.gettempdir()
                    os.makedirs(temp_dir, exist_ok=True)
                    output_path = os.path.join(temp_dir, f"doubao_seed3d_{task_id}.{file_format}")
                else:
                    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
                
                # Check if downloaded file is a zip file (Doubao API may return zip)
                content_type = file_response.headers.get('content-type', '')
                is_zip = 'zip' in content_type.lower() or result_url.endswith('.zip')
                
                if is_zip:
                    # If it's a zip file, extract and find target format file
                    print(f"[Doubao Seed3D] Detected zip file, extracting...")
                    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
                        tmp_zip.write(file_response.content)
                        tmp_zip_path = tmp_zip.name
                    
                    try:
                        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                            # Find target format file
                            target_ext = f'.{file_format}'
                            found_file = None
                            for file_info in zip_ref.namelist():
                                if file_info.endswith(target_ext):
                                    found_file = file_info
                                    break
                            
                            if found_file:
                                # Extract target file
                                with open(output_path, 'wb') as f:
                                    f.write(zip_ref.read(found_file))
                                print(f"[Doubao Seed3D] File extracted and saved: {output_path}")
                            else:
                                # If target format not found, extract all files
                                print(f"[Doubao Seed3D] {target_ext} file not found, extracting all files...")
                                zip_ref.extractall(os.path.dirname(output_path) or '.')
                                # Find extracted files
                                extracted_files = [f for f in os.listdir(os.path.dirname(output_path) or '.') if f.endswith(target_ext)]
                                if extracted_files:
                                    output_path = os.path.join(os.path.dirname(output_path) or '.', extracted_files[0])
                                    print(f"[Doubao Seed3D] File extracted and saved: {output_path}")
                                else:
                                    raise ValueError(f"Extracted files do not contain {target_ext} format file")
                    finally:
                        # Clean up temporary zip file
                        if os.path.exists(tmp_zip_path):
                            os.unlink(tmp_zip_path)
                else:
                    # Save file directly
                    with open(output_path, 'wb') as f:
                        f.write(file_response.content)
                    print(f"[Doubao Seed3D] File saved: {output_path}")
                
                return output_path
                
            elif status == 'failed' or status == 'error':
                error_msg = task_info.get('error') or task_info.get('message', 'Unknown error')
                raise RuntimeError(f"Task failed: {error_msg}")
                
            elif status in ['pending', 'processing', 'running', 'queued']:
                # Continue waiting
                time.sleep(poll_interval)
            else:
                # Unknown status, wait and retry
                print(f"[Doubao Seed3D] Warning: Unknown status '{status}', continuing to wait...")
                time.sleep(poll_interval)
                
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                error_msg = f"{error_msg}\nResponse details: {error_detail}"
            except:
                error_msg = f"{error_msg}\nResponse content: {e.response.text}"
        raise RuntimeError(f"Doubao Seed3D API call failed: {error_msg}")
    except Exception as e:
        print(f"Doubao Seed3D method failed: {e}")
        import traceback
        traceback.print_exc()
        raise

