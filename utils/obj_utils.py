"""
@FileName: dict_utils.py
@Description: 
@Author: HengLine
@Time: 2026/1/11 23:39
"""
from types import SimpleNamespace
from dataclasses import is_dataclass, asdict
from typing import Any, get_type_hints, get_origin, get_args, Dict, List, Union, Optional


class JSONObject:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, JSONObject(value))
            elif isinstance(value, list):
                setattr(self, key, [JSONObject(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)


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


def dict_to_obj(data: Any, clazz) -> Any:
    if data is None:
        return None

    # 处理基本类型：直接返回
    if not isinstance(data, (dict, list)):
        return data

    # 获取原始类型（剥离泛型）
    origin = get_origin(clazz)
    args = get_args(clazz)

    # 情况1: clazz 是 Dict[...] 或 dict
    if origin is dict or clazz is dict or (origin in (Dict, dict)):
        # Dict[K, V] → 我们只关心 value 类型（args[1]）
        key_type = args[0] if args else str
        value_type = args[1] if len(args) > 1 else Any
        return {k: dict_to_obj(v, value_type) for k, v in data.items()}

    # 情况2: clazz 是 List[...] 或 list
    if origin is list or clazz is list or (origin in (List, list)):
        item_type = args[0] if args else Any
        return [dict_to_obj(item, item_type) for item in data]

    # 情况3: clazz 是 Union（包括 Optional）
    if origin is Union:
        # Optional[T] == Union[T, None]
        # 尝试用第一个非-None 的类型解析
        non_none_types = [t for t in args if t is not type(None)]
        if non_none_types:
            return dict_to_obj(data, non_none_types[0])
        return data

    # 情况4: clazz 是一个普通类（如 Person, Address）
    if isinstance(clazz, type) and hasattr(clazz, '__init__'):
        if isinstance(data, dict):
            try:
                annotations = get_type_hints(clazz)
            except Exception:
                annotations = {}

            kwargs = {}
            for key, value in data.items():
                field_type = annotations.get(key, Any)
                kwargs[key] = dict_to_obj(value, field_type)
            return clazz(**kwargs)
        else:
            # 数据不是 dict，但目标是类？可能出错，回退
            return data

    # 默认：无法识别类型，原样返回
    return data



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


