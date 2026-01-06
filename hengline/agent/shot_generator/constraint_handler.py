"""
@FileName: constraint_handler.py
@Description: 约束处理器 - 处理和转换智能体3的连续性约束
@Author: HengLine
@Time: 2026/1/5 23:08
"""

import re
from enum import Enum
from typing import Dict, List, Any

from hengline.agent.continuity_guardian.continuity_guardian_model import HardConstraint


class ConstraintCategory(Enum):
    CHARACTER = "character"
    PROPS = "props"
    ENVIRONMENT = "environment"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    VISUAL = "visual"


class ConstraintHandler:
    """处理连续性约束"""

    def __init__(self):
        self.constraint_patterns = self._init_patterns()
        self.category_keywords = self._init_keywords()

    def _init_patterns(self) -> Dict[str, re.Pattern]:
        """初始化约束匹配模式"""
        return {
            "character_appearance": re.compile(
                r"(?:穿着|身着|穿戴|衣服|服装|发型|妆容)([^，。；]+)",
                re.UNICODE
            ),
            "prop_state": re.compile(
                r"(?:拿着|手持|握住|带着|携带|使用)([^，。；]+)",
                re.UNICODE
            ),
            "spatial_position": re.compile(
                r"(?:在|位于|处于|站在|坐在)([^，。；]+)",
                re.UNICODE
            ),
            "camera_angle": re.compile(
                r"(?:镜头|拍摄|角度|特写|近景|远景|中景)([^，。；]+)",
                re.UNICODE
            )
        }

    def _init_keywords(self) -> Dict[ConstraintCategory, List[str]]:
        """初始化分类关键词"""
        return {
            ConstraintCategory.CHARACTER: [
                "角色", "人物", "演员", "脸", "表情", "服装", "发型", "妆容",
                "character", "actor", "face", "expression", "clothing"
            ],
            ConstraintCategory.PROPS: [
                "道具", "物品", "武器", "工具", "杯子", "书", "手机",
                "prop", "object", "item", "weapon", "tool"
            ],
            ConstraintCategory.ENVIRONMENT: [
                "环境", "场景", "背景", "地点", "房间", "建筑", "自然",
                "environment", "scene", "background", "location"
            ],
            ConstraintCategory.SPATIAL: [
                "位置", "距离", "角度", "方向", "左边", "右边", "前面", "后面",
                "position", "distance", "angle", "direction"
            ]
        }

    def categorize_constraints(self, constraints: List[HardConstraint]) -> Dict[str, List[HardConstraint]]:
        """约束分类"""
        categorized = {}

        for constraint in constraints:
            category = self._determine_category(constraint)
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(constraint)

        return categorized

    def _determine_category(self, constraint: HardConstraint) -> str:
        """确定约束类别"""
        constraint_text = constraint.description.lower() + constraint.sora_instruction.lower()

        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in constraint_text:
                    return category.value

        # 默认基于类型判断
        type_mapping = {
            "character_appearance": "character",
            "prop_state": "props",
            "environment": "environment",
            "spatial": "spatial",
            "camera_angle": "visual"
        }
        return type_mapping.get(constraint.type, "visual")

    def extract_key_elements(self, constraints: List[HardConstraint]) -> Dict[str, Any]:
        """提取约束中的关键元素"""
        elements = {
            "characters": [],
            "props": [],
            "environments": [],
            "spatial_relations": [],
            "visual_requirements": []
        }

        for constraint in constraints:
            if constraint.type == "character_appearance":
                char_info = self._extract_character_info(constraint)
                if char_info:
                    elements["characters"].append(char_info)

            elif constraint.type == "prop_state":
                prop_info = self._extract_prop_info(constraint)
                if prop_info:
                    elements["props"].append(prop_info)

            elif constraint.type == "environment":
                env_info = self._extract_environment_info(constraint)
                if env_info:
                    elements["environments"].append(env_info)

            elif constraint.type == "spatial":
                spatial_info = self._extract_spatial_info(constraint)
                if spatial_info:
                    elements["spatial_relations"].append(spatial_info)

        return elements

    def _extract_character_info(self, constraint: HardConstraint) -> Dict[str, Any]:
        """提取角色信息"""
        info = {
            "name": "",
            "clothing": {},
            "posture": "",
            "expression": "",
            "accessories": []
        }

        # 从描述中提取角色名
        text = constraint.description
        name_pattern = re.compile(r"角色[：:]?\s*([^\s，。]+)")
        match = name_pattern.search(text)
        if match:
            info["name"] = match.group(1)

        # 提取服装信息
        clothing_pattern = re.compile(r"(?:穿着|身着|穿戴)([^，。；]+)")
        match = clothing_pattern.search(text)
        if match:
            clothing_text = match.group(1)
            info["clothing"] = self._parse_clothing_details(clothing_text)

        # 提取姿势和表情
        if "坐" in text:
            info["posture"] = "sitting"
        elif "站" in text:
            info["posture"] = "standing"
        elif "躺" in text:
            info["posture"] = "lying"

        if "微笑" in text:
            info["expression"] = "smiling"
        elif "严肃" in text:
            info["expression"] = "serious"
        elif "惊讶" in text:
            info["expression"] = "surprised"

        return info

    def _parse_clothing_details(self, clothing_text: str) -> Dict[str, str]:
        """解析服装细节"""
        details = {}

        # 提取颜色
        color_pattern = re.compile(r"(红色|蓝色|绿色|黄色|黑色|白色|灰色|棕色)")
        color_match = color_pattern.search(clothing_text)
        if color_match:
            details["color"] = color_match.group(1)

        # 提取服装类型
        type_keywords = ["衬衫", "T恤", "外套", "裤子", "裙子", "西装", "毛衣"]
        for keyword in type_keywords:
            if keyword in clothing_text:
                details["type"] = keyword
                break

        # 提取材质（如果有）
        material_keywords = ["棉质", "丝绸", "羊毛", "皮革", "牛仔"]
        for keyword in material_keywords:
            if keyword in clothing_text:
                details["material"] = keyword
                break

        return details

    def _extract_prop_info(self, constraint: HardConstraint) -> Dict[str, Any]:
        """提取道具信息"""
        info = {
            "name": "",
            "state": "",
            "location": "",
            "holder": ""
        }

        text = constraint.description

        # 提取道具名
        prop_pattern = re.compile(r"(?:咖啡杯|杯子|书|手机|包|钥匙|武器|工具)([^，。；]*)")
        match = prop_pattern.search(text)
        if match:
            info["name"] = match.group(0)

        # 提取状态
        if "半满" in text or "半空" in text:
            info["state"] = "half_full"
        elif "空" in text:
            info["state"] = "empty"
        elif "满" in text:
            info["state"] = "full"

        # 提取位置
        if "手中" in text:
            info["location"] = "in_hand"
        elif "桌上" in text:
            info["location"] = "on_table"
        elif "地上" in text:
            info["location"] = "on_floor"

        # 提取持有者
        holder_pattern = re.compile(r"([^\s，。]+)(?:的)?(?:手中|手里)")
        match = holder_pattern.search(text)
        if match:
            info["holder"] = match.group(1)

        return info

    def _extract_environment_info(self, constraint: HardConstraint) -> Dict[str, Any]:
        """提取环境信息"""
        info = {
            "type": "",
            "lighting": "",
            "time_of_day": "",
            "weather": "",
            "key_elements": []
        }

        text = constraint.description.lower()

        # 环境类型
        if "室内" in text or "房间" in text:
            info["type"] = "indoor"
            if "客厅" in text:
                info["type"] = "living_room"
            elif "卧室" in text:
                info["type"] = "bedroom"
            elif "办公室" in text:
                info["type"] = "office"
        elif "室外" in text or "户外" in text:
            info["type"] = "outdoor"

        # 光照条件
        if "阳光" in text or "日光" in text:
            info["lighting"] = "sunlight"
        elif "灯光" in text or "照明" in text:
            info["lighting"] = "artificial"
        elif "月光" in text:
            info["lighting"] = "moonlight"

        # 时间
        if "早晨" in text or "早上" in text:
            info["time_of_day"] = "morning"
        elif "中午" in text or "正午" in text:
            info["time_of_day"] = "noon"
        elif "下午" in text:
            info["time_of_day"] = "afternoon"
        elif "傍晚" in text or "黄昏" in text:
            info["time_of_day"] = "evening"
        elif "夜晚" in text or "晚上" in text:
            info["time_of_day"] = "night"

        return info

    def _extract_spatial_info(self, constraint: HardConstraint) -> Dict[str, Any]:
        """提取空间关系信息"""
        info = {
            "relation_type": "",
            "subject": "",
            "object": "",
            "distance": "",
            "direction": ""
        }

        text = constraint.description

        # 提取关系类型
        if "在...左边" in text or "左侧" in text:
            info["relation_type"] = "left_of"
        elif "在...右边" in text or "右侧" in text:
            info["relation_type"] = "right_of"
        elif "在...前面" in text or "前方" in text:
            info["relation_type"] = "in_front_of"
        elif "在...后面" in text or "后方" in text:
            info["relation_type"] = "behind"
        elif "靠近" in text or "接近" in text:
            info["relation_type"] = "near"
        elif "远离" in text or "距离" in text:
            info["relation_type"] = "far_from"

        return info

    def prioritize_constraints(self, constraints: List[HardConstraint]) -> List[HardConstraint]:
        """约束优先级排序"""
        # 第一级：基于优先级字段
        sorted_by_priority = sorted(constraints, key=lambda x: x.priority, reverse=True)

        # 第二级：基于类型的重要性
        type_weights = {
            "character_appearance": 100,
            "prop_state": 80,
            "environment": 60,
            "spatial": 70,
            "camera_angle": 90
        }

        def constraint_weight(constraint):
            base = type_weights.get(constraint.type, 50)
            return base + constraint.priority

        return sorted(sorted_by_priority, key=constraint_weight, reverse=True)

    def generate_constraint_summary(self, constraints: List[HardConstraint]) -> Dict[str, Any]:
        """生成约束摘要"""
        categorized = self.categorize_constraints(constraints)
        elements = self.extract_key_elements(constraints)

        summary = {
            "total_constraints": len(constraints),
            "by_category": {cat: len(cons) for cat, cons in categorized.items()},
            "by_priority": {
                "high": len([c for c in constraints if c.priority >= 8]),
                "medium": len([c for c in constraints if 5 <= c.priority < 8]),
                "low": len([c for c in constraints if c.priority < 5])
            },
            "key_characters": list(set([char["name"] for char in elements["characters"] if char["name"]])),
            "key_props": list(set([prop["name"] for prop in elements["props"] if prop["name"]])),
            "environments": list(set([env["type"] for env in elements["environments"] if env["type"]])),
            "critical_constraints": [
                {
                    "id": c.constraint_id,
                    "description": c.description,
                    "priority": c.priority
                }
                for c in constraints if c.priority >= 9
            ]
        }

        return summary
