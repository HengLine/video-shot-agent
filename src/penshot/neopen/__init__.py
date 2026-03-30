"""
@FileName: __init__.py
@Description: penshot 包初始化文件
@Author: HiPeng
@Github: https://github.com/neopen/video-shot-agent
@Time: 2025/10 - 2025/11
"""

# 导入主要模块
from penshot.logger import (debug, info, warning, error, critical)

from penshot.neopen.shot_config import ShotConfig

from penshot.neopen.shot_language import get_language


__all__ = [
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "get_language",
    "ShotConfig",
]

# 包版本
__version__ = "1.0.0"
