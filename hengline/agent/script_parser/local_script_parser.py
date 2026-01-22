"""
@FileName: local_script_parser.py
@Description: 
@Author: HengLine
@Time: 2026/1/10 22:33
"""
import re
from typing import Dict, Any

from hengline.agent.script_parser.base_script_parser import ScriptParser
from hengline.agent.script_parser.script_parser_model import UnifiedScript
from hengline.agent.workflow.workflow_models import ScriptType


class LocalScriptParser(ScriptParser):

    def __init__(self):
        """
        初始化剧本解析智能体
        """
        # 本地规则：用于校验和补全AI解析结果
        self.local_rules = {
            "character_name_patterns": [
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # 英文名
                r"([\u4e00-\u9fa5]{2,4})",  # 中文名（2-4字）
                r"(角色\s*[：:]\s*([^\s，。]+))",
                r"([^\s，。]+)\s*(?:说|道|问|喊|叫|称)"
            ],
            "scene_location_patterns": [
                r"(?:在|位于|处于)\s*([^，。]+?)(?:的|里|内|上|中)",
                r"(?:场景|地点)\s*[：:]\s*([^，。]+)",
                r"(?:INT\.|EXT\.)\s*([^-\n]+)",  # 室内/室外
                r"(?:室内|室外|房间|客厅|卧室|办公室|街道|公园)(?:[^，。]*?)"
            ],
            "dialogue_patterns": [
                r"([^\s，。：:]+)\s*[：:]\s*[\"']?([^\"'\n]+?)[\"']?[。！？]",
                r"([^\s，。]+)\s*(?:说|道|问|喊|叫|称)[：:]\s*[\"']?([^\"'\n]+?)[\"']?",
                r"[\"']([^\"'\n]+?)[\"']\s*[，,]?\s*([^\s，。]+)\s*(?:说|道)"
            ],
            "action_patterns": [
                r"([^\s，。]+)\s*(?:走|跑|坐|站|拿|看|笑|哭|转身|点头|摇头)(?:[^，。]*?)",
                r"(?:然后|接着|随后)\s*([^\s，。]+)\s*(?:开始|继续|停止)(?:[^，。]*?)",
                r"([^\s，。]+)\s*(?:手持|拿着|带着|使用)(?:[^，。]*?)"
            ]
        }

    def process(self, script_text: Any, unified_script: UnifiedScript) -> UnifiedScript:
        """
        应用本地规则进行校验和补全

        这是AI解析后的质量保证层
        """

        return unified_script
