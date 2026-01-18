"""
@FileName: file_utils.py
@Description: 
@Author: HengLine
@Time: 2026/1/18 12:24
"""
import json
from typing import Any

from utils.obj_utils import dict_to_obj


def load_from_json(json_path: str) -> str:
    """从JSON文件加载数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def load_from_obj(json_path: str, cls) -> Any:
    """从JSON文件加载数据"""
    return dict_to_obj(load_from_json(json_path), cls)


def save_to_json(cls: Any, file_name):
    # 保存为JSON
    with open(f"{file_name}.json", "w", encoding="utf-8") as f:
        json.dump(cls.to_dict(), f, ensure_ascii=False, indent=2)

