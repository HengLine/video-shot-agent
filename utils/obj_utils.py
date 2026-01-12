"""
@FileName: dict_utils.py
@Description: 
@Author: HengLine
@Time: 2026/1/11 23:39
"""
from typing import Any, Dict, List, Union
from dataclasses import is_dataclass, asdict


def obj_to_dict(obj: Any) -> Union[Dict, List, str, int, float, bool, None]:
    """
    安全地将任意对象转换为原生 Python 数据结构（dict/list/str/int...），
    适用于序列化、日志记录、JSON 输出等场景。

    支持：
      - dataclass
      - Pydantic v1 (BaseModel.dict())
      - Pydantic v2 (BaseModel.model_dump())
      - 普通对象（通过 __dict__）
      - 嵌套结构（递归）
      - 基本类型（直接返回）

    Args:
        obj: 任意 Python 对象

    Returns:
        可 JSON 序列化的原生数据结构
    """
    # None 或基本类型（str, int, float, bool）
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # 字典：递归处理 value
    if isinstance(obj, dict):
        return {k: obj_to_dict(v) for k, v in obj.items()}

    # 列表/元组：递归处理元素
    if isinstance(obj, (list, tuple)):
        return [obj_to_dict(item) for item in obj]

    # Pydantic v2 (LangChain 0.2+ 默认)
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            dumped = obj.model_dump()
            return obj_to_dict(dumped)  # 递归确保嵌套安全
        except Exception:
            pass  # fallback

    # Pydantic v1
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            dumped = obj.dict()
            return obj_to_dict(dumped)
        except Exception:
            pass  # fallback

    # dataclass
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return obj_to_dict(asdict(obj))
        except Exception:
            pass  # fallback

    # 普通对象（有 __dict__）
    if hasattr(obj, "__dict__"):
        try:
            return obj_to_dict(vars(obj))
        except Exception:
            pass

    # 最后手段：转为字符串（避免崩溃）
    return str(obj)


if __name__ == '__main__':
    from dataclasses import dataclass

    @dataclass
    class Person:
        name: str
        age: int

    # 示例 2: dataclass
    p = Person("Alice", 30)
    print(obj_to_dict(p))
    # → {'name': 'Alice', 'age': 30}

    # 示例 3: 嵌套结构
    data = {
        "user": p,
        "response": {'name': 'Alice', 'age': 30},
        "meta": ["a", {"b": Person("Bob", 25)}]
    }
    print(obj_to_dict(data))


