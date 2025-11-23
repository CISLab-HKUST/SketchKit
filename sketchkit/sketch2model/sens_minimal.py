"""
SENS (Sketch-based Neural Surface) Minimal Implementation
Sketch-to-3D model conversion based on pre-trained models
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import pickle
import requests
import zipfile
import tempfile
from typing import Union, Tuple, Optional

class MinimalSENSInference:
    """
    SENS最小推理实现
    
    这个类提供了SENS的核心功能，使用预训练模型将2D草图转换为3D模型，
    而不需要完整的SENS项目依赖。
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        初始化SENS推理器
        
        Args:
            model_path: 预训练模型路径，如果为None则使用默认路径
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.device = device
        self.model_path = model_path or self._get_default_model_path()
        self.model = None
        self.options = None
        
        # 加载模型
        self._load_model()
    
    def _get_default_model_path(self) -> str:
        """获取默认模型路径"""
        # 优先使用本地预训练模型
        current_file = Path(__file__).resolve()
        local_paths = [
            str(current_file.parent / "external" / "sens" / "assets" / "checkpoints"),
            str(current_file.parent.parent / "models" / "external" / "sens" / "assets" / "checkpoints"),
            "assets/checkpoints",
            "checkpoints",
            str(Path.home() / ".cache" / "sketchkit" / "sens_models" / "checkpoints"),
        ]
        
        for path in local_paths:
            if os.path.exists(path):
                return path
        
        # 如果本地没有，返回Google Drive链接
        return "https://drive.google.com/file/d/15b98DQL3ebL_SaccRys0zJimCxppxaVo/view?usp=sharing"
    
    def _load_model(self):
        """加载预训练模型"""
        try:
            # 检查是否需要下载模型
            if self.model_path.startswith("http"):
                self.model_path = self._download_pretrained_model()
            
            # 添加SENS项目路径（多个可能的位置）
            current_file = Path(__file__).resolve()
            possible_sens_paths = [
                current_file.parent / "external" / "sens",  # 从sketch2model目录
                current_file.parent.parent / "external" / "sens",  # 从sketchkit目录
                current_file.parent.parent.parent / "external" / "sens",  # 从项目根目录
            ]
            
            sens_path = None
            for path in possible_sens_paths:
                if path.exists() and (path / "options.py").exists():
                    sens_path = path
                    break
            
            if sens_path is None:
                # 尝试从模型目录中查找（解压后的模型可能包含SENS代码）
                model_path_obj = Path(self.model_path)
                if model_path_obj.exists():
                    # 检查模型目录的父目录
                    parent_dirs = [
                        model_path_obj.parent.parent,  # 从checkpoints往上
                        model_path_obj.parent.parent.parent,  # 再往上
                    ]
                    for parent in parent_dirs:
                        test_path = parent / "options.py"
                        if test_path.exists():
                            sens_path = parent
                            break
            
            if sens_path is None:
                raise ImportError(
                    "无法找到SENS项目目录。请确保SENS项目位于以下位置之一：\n"
                    f"  - {Path(__file__).parent / 'external' / 'sens'}\n"
                    f"  - 或模型解压目录包含SENS源代码"
                )
            
            # 添加路径到sys.path
            if str(sens_path) not in sys.path:
                sys.path.insert(0, str(sens_path))
            ui_sketch_path = sens_path / "ui_sketch"
            if ui_sketch_path.exists() and str(ui_sketch_path) not in sys.path:
                sys.path.insert(0, str(ui_sketch_path))
            
            # 导入SENS相关模块
            from options import SketchOptions
            from ui_sketch.sketch_inference import SketchInference
            
            # 初始化选项
            self.options = SketchOptions()
            self.options.tags = ['chairs', 'chairs_sym_hard']
            
            # 修复dataset_size问题：推理时不需要dataset_size，设置为一个合理的默认值
            # 我们需要在SketchInference创建之前修复spaghetti模型的options
            # 因为SketchInference会在__init__中调用Options(tag=opt.spaghetti_tag).load()
            try:
                from options import Options
                from ui_sketch import occ_inference
                
                # 创建一个包装类来拦截options加载并修复dataset_size
                class FixedOptions(Options):
                    def load(self):
                        loaded = super().load()
                        # 如果加载的options中dataset_size是-1，修复它
                        if hasattr(loaded, 'dataset_size') and loaded.dataset_size == -1:
                            print(f"修复spaghetti模型的dataset_size: -1 -> 1000")
                            loaded.dataset_size = 1000
                            # 保存修复后的options
                            loaded.save()
                            print(f"已保存修复后的options到: {loaded.save_path}")
                        return loaded
                
                # 临时替换Options类，让SketchInference和occ_inference使用修复后的版本
                import ui_sketch.sketch_inference as sketch_inference_module
                import ui_sketch.occ_inference as occ_inference_module
                
                original_options_sketch = getattr(sketch_inference_module, 'Options', None)
                original_options_occ = getattr(occ_inference_module, 'Options', None)
                
                sketch_inference_module.Options = FixedOptions
                if hasattr(occ_inference_module, 'Options'):
                    occ_inference_module.Options = FixedOptions
                
                try:
                    # 创建推理器
                    self.model = SketchInference(self.options)
                finally:
                    # 恢复原始的Options类
                    if original_options_sketch:
                        sketch_inference_module.Options = original_options_sketch
                    if original_options_occ:
                        occ_inference_module.Options = original_options_occ
            except Exception as e:
                print(f"警告: 使用备用方法修复dataset_size: {e}")
                import traceback
                traceback.print_exc()
                # 备用方法：直接修改SketchInference的代码路径
                # 先加载并修复options文件
                try:
                    from options import Options
                    spaghetti_opt = Options(tag=self.options.spaghetti_tag)
                    loaded_opt = spaghetti_opt.load()
                    if loaded_opt and hasattr(loaded_opt, 'dataset_size') and loaded_opt.dataset_size == -1:
                        loaded_opt.dataset_size = 1000
                        loaded_opt.save()
                        print(f"已修复并保存options到: {loaded_opt.save_path}")
                except Exception as e2:
                    print(f"警告: 无法修复options文件: {e2}")
                
                # 创建推理器
                self.model = SketchInference(self.options)
            
            print(f"SENS模型加载成功，使用设备: {self.device}")
            print(f"模型路径: {self.model_path}")
            
        except ImportError as e:
            print(f"错误: 无法导入SENS模块: {e}")
            raise RuntimeError(f"SENS模块导入失败: {e}")
        except Exception as e:
            print(f"错误: 模型加载失败: {e}")
            raise RuntimeError(f"SENS模型加载失败: {e}")
    
    def _download_pretrained_model(self) -> str:
        """从Google Drive下载预训练模型"""
        print("正在从Google Drive下载SENS预训练模型...")
        
        # Google Drive文件ID
        file_id = "15b98DQL3ebL_SaccRys0zJimCxppxaVo"
        
        # 创建模型缓存目录
        cache_dir = Path.home() / ".cache" / "sketchkit" / "sens_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已经下载（检查checkpoints目录是否存在且有内容）
        model_dir = cache_dir / "checkpoints"
        if model_dir.exists():
            # 检查是否有任何文件或子目录（排除隐藏文件）
            try:
                items = [item for item in model_dir.iterdir() if not item.name.startswith('.')]
                if items:
                    # 进一步检查是否有模型文件（.pth, .pkl等）
                    model_files = [item for item in items if item.suffix in ['.pth', '.pkl', '.pt', '.ckpt'] or item.is_dir()]
                    if model_files:
                        print(f"使用缓存的模型: {model_dir}")
                        return str(model_dir)
            except Exception as e:
                # 如果检查失败，继续下载流程
                pass
        
        try:
            # 尝试使用 gdown 库（如果可用）
            try:
                import gdown
                print("使用 gdown 下载模型...")
                url = f"https://drive.google.com/uc?id={file_id}"
                output_zip = str(cache_dir / "sens_model.zip")
                gdown.download(url, output_zip, quiet=False)
                
                # 解压文件
                print("正在解压模型文件...")
                with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                    zip_ref.extractall(cache_dir)
                
                # 清理zip文件
                os.unlink(output_zip)
                
                print(f"SENS预训练模型下载完成: {model_dir}")
                return str(model_dir)
            except ImportError:
                print("gdown 未安装，尝试使用 requests 下载...")
                # 如果没有 gdown，使用 requests（需要处理 Google Drive 的确认页面）
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                
                session = requests.Session()
                response = session.get(download_url, stream=True)
                response.raise_for_status()
                
                # 检查是否是确认页面（大文件需要确认）
                if 'virus scan warning' in response.text.lower() or 'download anyway' in response.text.lower():
                    # 提取确认token
                    import re
                    confirm_match = re.search(r'confirm=([^&]+)', response.text)
                    if confirm_match:
                        confirm_token = confirm_match.group(1)
                        download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                        response = session.get(download_url, stream=True)
                        response.raise_for_status()
                
                # 检查响应内容类型
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type:
                    print("错误: Google Drive 返回了HTML页面而不是文件")
                    print("提示: 请手动下载模型或安装 gdown: pip install gdown")
                    print(f"下载链接: https://drive.google.com/file/d/{file_id}/view")
                    raise RuntimeError("无法自动下载，请手动下载模型或安装 gdown")
                
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                    print(f"下载到临时文件: {tmp_file.name}")
                    
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\r下载进度: {progress:.1f}%", end="", flush=True)
                    
                    print("\n下载完成!")
                
                # 验证文件是否为有效的zip文件
                print("正在验证下载的文件...")
                try:
                    with zipfile.ZipFile(tmp_file.name, 'r') as test_zip:
                        test_zip.testzip()
                except zipfile.BadZipFile:
                    print(f"错误: 下载的文件不是有效的zip文件")
                    print("提示: 请手动下载模型或安装 gdown: pip install gdown")
                    print(f"下载链接: https://drive.google.com/file/d/{file_id}/view")
                    os.unlink(tmp_file.name)
                    raise RuntimeError("下载的文件无效，请手动下载模型或安装 gdown")
                
                # 解压文件
                print("正在解压模型文件...")
                with zipfile.ZipFile(tmp_file.name, 'r') as zip_ref:
                    zip_ref.extractall(cache_dir)
                
                # 清理临时文件
                os.unlink(tmp_file.name)
                
                print(f"SENS预训练模型下载完成: {model_dir}")
                return str(model_dir)
            
        except Exception as e:
            print(f"模型下载失败: {e}")
            print(f"提示: 请手动下载模型到 {cache_dir}")
            print(f"下载链接: https://drive.google.com/file/d/{file_id}/view")
            print(f"或安装 gdown: pip install gdown")
            raise RuntimeError(f"无法下载SENS预训练模型: {e}")
    
    def preprocess_sketch(self, sketch_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        预处理草图输入
        
        Args:
            sketch_input: 草图输入，可以是文件路径、numpy数组或PIL图像
            
        Returns:
            预处理后的numpy数组 (H, W, 1) 格式，以匹配prepare_real_sketches的期望
        """
        # 处理不同类型的输入
        if isinstance(sketch_input, str):
            # 文件路径
            image = Image.open(sketch_input).convert('L')
        elif isinstance(sketch_input, np.ndarray):
            # numpy数组
            if sketch_input.ndim == 3:
                image = Image.fromarray(sketch_input).convert('L')
            else:
                image = Image.fromarray(sketch_input, mode='L')
        elif isinstance(sketch_input, Image.Image):
            # PIL图像
            image = sketch_input.convert('L')
        else:
            raise ValueError("不支持的输入类型")
        
        # 调整大小到256x256
        image = image.resize((256, 256), Image.Resampling.LANCZOS)
        
        # 转换为numpy数组并归一化
        # prepare_real_sketches期望输入是(H, W, C)格式，需要添加通道维度
        sketch_array = np.array(image, dtype=np.float32) / 255.0
        # 添加通道维度: (H, W) -> (H, W, 1)
        if sketch_array.ndim == 2:
            sketch_array = sketch_array[:, :, np.newaxis]
        
        return sketch_array
    
    def infer(self, sketch_input: Union[str, np.ndarray, Image.Image], 
              output_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行SENS推理
        
        Args:
            sketch_input: 草图输入
            output_path: 输出OBJ文件路径，如果为None则不保存文件
            
        Returns:
            (vertices, faces): 3D模型的顶点和面
        """
        # 预处理草图
        sketch_array = self.preprocess_sketch(sketch_input)
        
        # 使用完整的SENS模型
        if self.model is None:
            raise RuntimeError("SENS模型未正确加载，请检查模型路径和依赖")
        
        try:
            result = self.model.sketch2mesh(sketch_array)
            
            # 处理返回值（可能是tuple或其他格式）
            import torch
            if isinstance(result, tuple):
                vertices, faces = result
            else:
                # 如果不是tuple，尝试解包
                vertices, faces = result[0], result[1]
            
            # 转换为numpy数组（处理torch tensor，包括嵌套结构）
            def to_numpy(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
                elif hasattr(x, 'cpu'):
                    try:
                        return x.cpu().numpy()
                    except:
                        return x.detach().cpu().numpy()
                elif hasattr(x, 'numpy'):
                    return x.numpy()
                elif isinstance(x, (list, tuple)):
                    return [to_numpy(item) for item in x]
                else:
                    try:
                        return np.asarray(x)
                    except:
                        # 如果是其他类型，尝试直接返回
                        return x
            
            # 深度转换（处理嵌套tuple/list）
            vertices = to_numpy(vertices)
            faces = to_numpy(faces)
            
            # 确保是numpy数组
            if isinstance(vertices, (list, tuple)):
                vertices = np.asarray(vertices[0] if vertices else vertices)
            if isinstance(faces, (list, tuple)):
                faces = np.asarray(faces[0] if faces else faces)
            
            print(f"SENS推理成功: {len(vertices)} 顶点, {len(faces)} 面")
        except Exception as e:
            print(f"SENS推理失败: {e}")
            raise RuntimeError(f"SENS推理执行失败: {e}")
        
        # 保存OBJ文件
        if output_path:
            self._save_obj(vertices, faces, output_path)
            print(f"3D模型已保存到: {output_path}")
        
        return vertices, faces
    
    
    def _save_obj(self, vertices, faces, output_path: str):
        """保存OBJ文件"""
        # 确保是numpy数组（可能是torch tensor）
        import torch
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.detach().cpu().numpy()
        elif hasattr(vertices, 'cpu'):
            vertices = vertices.cpu().numpy()
        elif hasattr(vertices, 'numpy'):
            vertices = vertices.numpy()
        vertices = np.asarray(vertices)
        
        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()
        elif hasattr(faces, 'cpu'):
            faces = faces.cpu().numpy()
        elif hasattr(faces, 'numpy'):
            faces = faces.numpy()
        faces = np.asarray(faces)
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            # 写入顶点
            for v in vertices:
                v_flat = np.asarray(v).flatten()
                if len(v_flat) >= 3:
                    f.write(f"v {float(v_flat[0]):.6f} {float(v_flat[1]):.6f} {float(v_flat[2]):.6f}\n")
            
            # 写入面（OBJ格式从1开始索引）
            for face in faces:
                face_flat = np.asarray(face).flatten()
                if len(face_flat) >= 3:
                    f.write(f"f {int(face_flat[0])+1} {int(face_flat[1])+1} {int(face_flat[2])+1}\n")


# 便利函数
def sketch_to_3d_sens(sketch_input: Union[str, np.ndarray, Image.Image], 
                      output_path: Optional[str] = None,
                      model_path: Optional[str] = None,
                      device: str = 'cpu') -> str:
    """
    便利函数：将草图转换为3D模型
    
    Args:
        sketch_input: 草图输入
        output_path: 输出OBJ文件路径（如果为None，会自动生成）
        model_path: 模型路径
        device: 计算设备
        
    Returns:
        str: 输出OBJ文件路径
    """
    inference = MinimalSENSInference(model_path=model_path, device=device)
    
    # 如果没有指定输出路径，生成一个临时路径
    if output_path is None:
        import tempfile
        temp_dir = tempfile.gettempdir()
        import time
        timestamp = int(time.time() * 1000)
        output_path = os.path.join(temp_dir, f"sens_output_{timestamp}.obj")
    
    # 执行推理并保存文件
    inference.infer(sketch_input, output_path)
    
    # 返回文件路径
    return output_path


if __name__ == "__main__":
    # 测试代码
    print("SENS最小实现测试")
    
    # 创建测试图像
    test_image = np.ones((256, 256), dtype=np.uint8) * 255
    test_image[50:200, 50:200] = 0  # 黑色矩形
    
    # 测试推理
    inference = MinimalSENSInference()
    vertices, faces = inference.infer(test_image, "test_sens_output.obj")
    
    print(f"生成模型: {len(vertices)} 顶点, {len(faces)} 面")
