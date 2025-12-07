"""
@FileName: workflow_models.py
@Description:  工作流模型定义模块
@Author: HengLine
@Time: 2025/11/30 19:12
"""
from enum import Enum, unique


@unique
class VideoStyle(Enum):
    # 逼真
    REALISTIC = 'realistic'
    # 动漫
    ANIME = 'anime'
    # 电影
    CINEMATIC = 'cinematic'
    # 卡通
    CARTOON = 'cartoon'
