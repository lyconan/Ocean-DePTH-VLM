"""模块加载器 — 解决数字开头文件名无法用import导入的问题

由于脚本文件名以数字开头（如 07_extract_clip_embeddings.py），
Python的import语句无法直接导入。此模块提供统一的动态加载接口。
"""
import importlib
import importlib.util
import sys
from pathlib import Path


_SCRIPTS_DIR = Path(__file__).parent.parent


def load_script_module(script_name: str):
    """动态加载scripts目录下的脚本模块。
    
    Args:
        script_name: 脚本文件名（不含.py后缀），如 '07_extract_clip_embeddings'
                    或 '10_graphrag_pipeline'
    
    Returns:
        加载的模块对象
    
    Example:
        >>> mod = load_script_module('07_extract_clip_embeddings')
        >>> emb = mod.extract_single_clip_embedding('image.png')
    """
    module_path = _SCRIPTS_DIR / f"{script_name}.py"
    
    if not module_path.exists():
        raise FileNotFoundError(f"脚本文件不存在: {module_path}")
    
    # 生成合法的模块名（替换数字开头）
    spec_name = f"_depth_vlm_{script_name}"
    
    if spec_name in sys.modules:
        return sys.modules[spec_name]
    
    spec = importlib.util.spec_from_file_location(spec_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_path}")

    module = importlib.util.module_from_spec(spec)
    # 先执行模块，成功后再缓存，避免exec_module失败导致缓存中毒
    spec.loader.exec_module(module)
    sys.modules[spec_name] = module

    return module
