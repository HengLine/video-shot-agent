"""
@FileName: RuleScriptParser.py
@Description: 
@Author: HengLine
@Time: 2026/1/26 14:38
"""
from typing import Any

from hengline.agent.script_parser.base_script_parser import BaseScriptParser
from hengline.agent.script_parser.script_parser_models import ParsedScript
from hengline.agent.workflow.workflow_models import ScriptType


class RuleScriptParser(BaseScriptParser):

    def __init__(self):
        """
        初始化剧本解析智能体

        """

    def parser(self, script_text: Any, script_format: ScriptType) -> ParsedScript | None:
        pass
