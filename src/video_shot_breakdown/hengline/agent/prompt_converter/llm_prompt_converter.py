"""
@FileName: llm_prompt_converter.py
@Description: 基于LLM的提示词转换器 - 从ParsedScript获取完整上下文
@Author: HengLine
@Github: https://github.com/HengLine/video-shot-agent
@Time: 2026/1/26 23:36
"""
from typing import Optional

from video_shot_breakdown.hengline.agent.base_agent import BaseAgent
from video_shot_breakdown.hengline.agent.prompt_converter.base_prompt_converter import BasePromptConverter
from video_shot_breakdown.hengline.agent.prompt_converter.prompt_converter_models import AIVideoPrompt, AIVideoInstructions
from video_shot_breakdown.hengline.agent.prompt_converter.template_prompt_converter import TemplatePromptConverter
from video_shot_breakdown.hengline.agent.script_parser.script_parser_models import ParsedScript
from video_shot_breakdown.hengline.agent.video_splitter.video_splitter_models import FragmentSequence, VideoFragment
from video_shot_breakdown.hengline.hengline_config import HengLineConfig
from video_shot_breakdown.hengline.language_manage import get_language
from video_shot_breakdown.logger import info, error


class LLMPromptConverter(BasePromptConverter, BaseAgent):
    """基于LLM的提示词转换器 - 从ParsedScript获取完整上下文"""

    def __init__(self, llm_client, config: Optional[HengLineConfig]):
        super().__init__(config)
        self.llm_client = llm_client
        self.parsed_script = None
        self.global_metadata = None

    def convert(self, fragment_sequence: FragmentSequence, parsed_script: ParsedScript) -> AIVideoInstructions:
        """使用LLM转换提示词 - 传入ParsedScript获取完整上下文"""
        info(f"使用LLM转换提示词，片段数: {len(fragment_sequence.fragments)}")

        # 保存ParsedScript供后续使用
        self.parsed_script = parsed_script
        self.global_metadata = parsed_script.global_metadata

        prompts = []
        project_info = {
            "title": fragment_sequence.source_info.get("title", "AI视频项目"),
            "total_fragments": fragment_sequence.stats.get("fragment_count", 0),
            "total_duration": fragment_sequence.stats.get("total_duration", 0.0)
        }

        for fragment in fragment_sequence.fragments:
            try:
                prompt = self._convert_fragment_with_llm(fragment)
                prompts.append(prompt)
            except Exception as e:
                error(f"片段{fragment.id}转换失败: {str(e)}")
                # 降级到模板转换
                template_converter = TemplatePromptConverter(self.config)
                fallback_prompt = template_converter.convert_fragment(fragment)
                prompts.append(fallback_prompt)

        instructions = AIVideoInstructions(
            project_info=project_info,
            fragments=prompts
        )
        return self.post_process(instructions)

    def _convert_fragment_with_llm(self, fragment: VideoFragment) -> AIVideoPrompt:
        """使用LLM转换单个片段 - 增强版，从ParsedScript获取完整上下文"""

        # 检测原始剧本语言
        original_language = self._detect_original_language(fragment)

        # 获取当前片段对应的场景和元素信息
        scene_info = self._get_scene_info_for_fragment(fragment)
        element_info = self._get_element_info_for_fragment(fragment)

        # 格式化全局上下文（简洁格式用于prompt_converter）
        global_context = self._format_global_context(self.global_metadata)

        # 获取完整剧本上下文
        full_script_context = self._get_full_script_context(fragment)

        # 准备提示词
        user_prompt = self._get_prompt_template("prompt_converter_user")

        prompt_template = user_prompt.format(
            fragment_id=fragment.id,
            description=fragment.description,
            duration=fragment.duration,
            character=fragment.continuity_notes.get("main_character", ""),
            location=fragment.continuity_notes.get("location", ""),
            original_language=original_language,
            dm_model=self.config.target_model.value,
            video_style=self.config.default_style.value,
            max_length=self.config.max_prompt_length,
            min_length=self.config.min_prompt_length,
            # 新增：传递增强的上下文信息
            global_context=global_context,
            scene_info=scene_info,
            element_info=element_info,
            full_script_context=full_script_context
        )

        # 调用LLM
        system_prompt = self._get_prompt_template("prompt_converter_system")
        result = self._call_llm_parse_with_retry(self.llm_client, system_prompt, prompt_template)

        # 获取生成的提示词
        english_prompt = result.get("prompt", "")
        original_prompt = result.get("original_prompt", "")

        # 合并双语提示词：英文在前，原始语言在后
        combined_prompt = f"{english_prompt}\n\n{original_prompt}"

        return AIVideoPrompt(
            fragment_id=fragment.id,
            prompt=combined_prompt,
            negative_prompt=result.get("negative_prompt", self.config.default_negative_prompt),
            duration=fragment.duration,
            model=self.config.target_model.value,
            style=result.get("style_hint")
        )

    def _get_scene_info_for_fragment(self, fragment: VideoFragment) -> str:
        """获取片段对应的场景信息"""
        if not self.parsed_script:
            return ""

        # 从fragment的shot_id中提取场景ID
        shot_id = fragment.shot_id
        scene_id = None

        # 遍历所有场景，找到包含该镜头的场景
        for scene in self.parsed_script.scenes:
            if hasattr(scene, 'elements') and scene.elements:
                # 这里需要根据实际情况匹配shot_id和element_id的关系
                # 简化处理：从continuity_notes中获取location
                if "location" in fragment.continuity_notes:
                    location = fragment.continuity_notes["location"]
                    if location in scene.location or scene.id in location:
                        scene_id = scene.id
                        break

        if not scene_id:
            return ""

        for scene in self.parsed_script.scenes:
            if scene.id == scene_id:
                weather = getattr(scene, 'weather', '未知')
                time_of_day = getattr(scene, 'time_of_day', '未知')
                characters = []
                for elem in scene.elements:
                    if elem.character and elem.character not in characters:
                        characters.append(elem.character)

                return f"当前场景：{scene.location}，天气：{weather}，时间：{time_of_day}，角色：{', '.join(characters)}"
        return ""

    def _get_element_info_for_fragment(self, fragment: VideoFragment) -> str:
        """获取片段对应的元素信息（特别是台词）"""
        if not self.parsed_script:
            return ""

        # 从fragment的metadata中获取original_element_ids
        element_ids = []
        if hasattr(fragment, 'metadata') and fragment.metadata:
            element_ids = fragment.metadata.get("element_ids", [])

        if not element_ids:
            return ""

        # 收集所有相关元素的完整内容
        elements_text = []
        for scene in self.parsed_script.scenes:
            for elem in scene.elements:
                if elem.id in element_ids:
                    elem_type = "对话" if elem.type == "dialogue" else "动作" if elem.type == "action" else "场景"
                    char_info = f"（角色：{elem.character}）" if elem.character else ""
                    content = elem.content if elem.content else elem.description
                    elements_text.append(f"  - [{elem_type}{char_info}] {content}")

        if elements_text:
            return "本片段包含的原始元素：\n" + "\n".join(elements_text)
        return ""

    def _get_full_script_context(self, fragment: VideoFragment) -> str:
        """获取完整的剧本上下文，帮助LLM理解叙事顺序"""
        if not self.parsed_script:
            return ""

        # 构建完整的剧本时间线摘要
        timeline = []
        for scene_idx, scene in enumerate(self.parsed_script.scenes):
            scene_desc = f"场景{scene_idx + 1}：{scene.location}，{getattr(scene, 'time_of_day', '未知')}，{getattr(scene, 'weather', '未知')}"
            key_elements = []
            for elem in scene.elements[:3]:  # 只取前3个关键元素
                if elem.type == "dialogue" and elem.content:
                    key_elements.append(f"「{elem.character}：{elem.content[:30]}」")
                elif elem.type == "action" and elem.description:
                    key_elements.append(f"[{elem.description[:30]}]")
            if key_elements:
                scene_desc += "：" + "；".join(key_elements)
            timeline.append(scene_desc)

        return "完整剧本时间线：\n" + "\n".join(timeline)

    def _detect_original_language(self, fragment: VideoFragment) -> str:
        """检测原始剧本语言"""
        if hasattr(fragment, 'metadata') and fragment.metadata:
            return fragment.metadata.get("original_language", "zh")
        return get_language().value
