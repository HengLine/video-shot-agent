"""
@FileName: action_uration_tool.py
@Description: 动作时长估算算法
@Author: HengLine
@Time: 2025/10/24 14:01
"""
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import jieba
import yaml

# 全局配置锁
_config_lock = threading.RLock()
_current_config: Optional[Dict[str, Any]] = None
_config_path: Optional[Path] = None


class ActionDurationConfig:
    """
    生产级动作时长估算器（修复版）
    - 对话时长 = max(字数 × 情绪因子, min_duration)
    - 角色因子仅在顶层应用一次
    - 动作/对话内部逻辑与角色完全解耦合
    """

    def __init__(self, config_path: str = "action_duration_config.yaml"):
        global _current_config, _config_path
        self.config_path = Path(__file__).parent / "zh" / config_path

        with _config_lock:
            if _current_config is None or _config_path != self.config_path:
                self._load_config()
                _config_path = self.config_path

        self._init_jieba()

    def _load_config(self):
        """加载 YAML 配置（深拷贝防污染）"""
        global _current_config
        with _config_lock:
            with open(self.config_path, "r", encoding="utf-8") as f:
                import copy
                _current_config = copy.deepcopy(yaml.safe_load(f))

    def _init_jieba(self):
        """优化中文分词"""
        if _current_config:
            for verb in _current_config.get("base_actions", {}):
                jieba.add_word(verb, freq=2000, tag='v')
            for mod in _current_config.get("modifiers", {}):
                jieba.add_word(mod, freq=2000, tag='d')

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        global _current_config
        if _current_config is None:
            self._load_config()

        return _current_config

    @classmethod
    def reload_config(cls, config_path: str = "action_duration_config.yaml"):
        """热重载配置"""
        global _current_config, _config_path
        with _config_lock:
            config_path = Path(__file__).parent / "zh" / config_path
            with open(config_path, "r", encoding="utf-8") as f:
                import copy
                _current_config = copy.deepcopy(yaml.safe_load(f))
            _config_path = Path(config_path)
        # 此处建议由调用方管理实例生命周期
        # cls.clear_cache()
