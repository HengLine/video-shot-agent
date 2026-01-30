"""
@FileName: script_parser_agent.py
@Description: LLM 剧本解析基类，包含复杂度评估和路由决策逻辑
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from abc import abstractmethod, ABC
from datetime import datetime
from typing import Dict, Any

from video_shot_breakdown.hengline.agent.base_models import ScriptType, ElementType
from video_shot_breakdown.hengline.agent.script_parser.script_parser_models import ParsedScript, SceneInfo, CharacterInfo, BaseElement
from video_shot_breakdown.hengline.logger import info, warning
from video_shot_breakdown.hengline.tools.script_assessor_tool import ComplexityAssessor


class BaseScriptParser(ABC):
    """优化版剧本解析智能体"""

    def __post_init__(self):
        """
        初始化剧本解析智能体
        """
        self.complexity_assessor = ComplexityAssessor()

    @abstractmethod
    def parser(self, script_text: Any, script_format: ScriptType) -> ParsedScript:
        """处理输入数据（子类实现）"""
        raise NotImplementedError("子类必须实现process方法")

    def post_process(self, parsed_script: ParsedScript) -> ParsedScript:
        """后处理：填充统计数据等"""
        # 计算统计数据
        total_elements = 0
        total_duration = 0.0
        dialogue_count = 0
        action_count = 0

        for scene in parsed_script.scenes:
            total_elements += len(scene.elements)
            for element in scene.elements:
                total_duration += element.duration
                if element.type == ElementType.DIALOGUE:
                    dialogue_count += 1
                elif element.type == ElementType.ACTION:
                    action_count += 1

        # 更新统计数据
        parsed_script.stats.update({
            "total_elements": total_elements,
            "total_duration": round(total_duration, 2),
            "dialogue_count": dialogue_count,
            "action_count": action_count
        })

        # 更新元数据
        if not parsed_script.metadata.get("parser_type"):
            parsed_script.metadata["parser_type"] = self.__class__.__name__

        info(f"解析完成: {total_elements}个元素, {len(parsed_script.scenes)}个场景")
        return parsed_script

    def validate_parsed_result(self, parsed_script: ParsedScript) -> bool:
        """验证解析结果的基本有效性"""
        if not parsed_script.scenes:
            warning("解析结果为空：没有找到任何场景")
            return False

        # 检查元素顺序连续性
        all_elements = []
        for scene in parsed_script.scenes:
            all_elements.extend(scene.elements)

        if all_elements:
            sequences = [elem.sequence for elem in all_elements]
            if sorted(sequences) != list(range(1, len(sequences) + 1)):
                warning(f"元素顺序不连续: {sequences}")

        return True

    def _generate_element_id(self, scene_idx: int, elem_idx: int) -> str:
        """生成元素ID"""
        return f"elem_{scene_idx + 1:03d}_{elem_idx + 1:03d}"

    def _generate_scene_id(self, scene_idx: int) -> str:
        """生成场景ID"""
        return f"scene_{scene_idx + 1:03d}"

    def _build_parsed_script(self, data: Dict[str, Any]) -> ParsedScript:
        """构建解析结果对象"""
        # 构建场景列表
        scenes = []
        for scene_idx, scene_data in enumerate(data.get("scenes", [])):
            elements = []
            for elem_idx, elem_data in enumerate(scene_data.get("elements", [])):
                # 确保元素类型是ElementType枚举
                elem_type = ElementType(elem_data.get("type", "action"))

                element = BaseElement(
                    id=elem_data.get("id", self._generate_element_id(scene_idx, elem_idx)),
                    type=elem_type,
                    sequence=elem_data.get("sequence", elem_idx + 1),
                    duration=elem_data.get("duration", 3.0),
                    confidence=elem_data.get("confidence", 0.8),
                    content=elem_data.get("content", ""),
                    character=elem_data.get("character"),
                    target_character=elem_data.get("target_character"),
                    description=elem_data.get("description", ""),
                    intensity=elem_data.get("intensity", 0.5),
                    emotion=elem_data.get("emotion", "自然")
                )
                elements.append(element)

            scene = SceneInfo(
                id=scene_data.get("id", self._generate_scene_id(scene_idx)),
                location=scene_data.get("location", "未知地点"),
                description=scene_data.get("description"),
                time_of_day=scene_data.get("time_of_day"),
                elements=elements
            )
            scenes.append(scene)

        # 构建角色列表
        characters = [
            CharacterInfo(
                name=char_data["name"],
                gender=char_data["gender"],
                role=char_data["role"],
                description=char_data.get("description"),
                key_traits=char_data.get("key_traits", [])
            )
            for char_data in data.get("characters", [])
        ]

        # 返回完整解析结果
        return ParsedScript(
            title=data.get("title"),
            characters=characters,
            scenes=scenes,
            metadata={
                "parsed_at": datetime.now().isoformat(),
                "version": "mvp_1.0",
                "source_type": "llm_parsed",
                "parser_type": self.__class__.__name__
            }
        )
