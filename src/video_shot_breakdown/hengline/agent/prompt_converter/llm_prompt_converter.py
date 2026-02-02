"""
@FileName: llm_prompt_converter.py
@Description: 
@Author: HengLine
@Time: 2026/1/26 23:36
"""
from typing import Optional

from video_shot_breakdown.hengline.agent.base_agent import BaseAgent
from video_shot_breakdown.hengline.agent.prompt_converter.base_prompt_converter import BasePromptConverter
from video_shot_breakdown.hengline.agent.prompt_converter.prompt_converter_models import AIVideoPrompt, AIVideoInstructions
from video_shot_breakdown.hengline.agent.prompt_converter.template_prompt_converter import TemplatePromptConverter
from video_shot_breakdown.hengline.agent.video_splitter.video_splitter_models import FragmentSequence, VideoFragment
from video_shot_breakdown.hengline.hengline_config import HengLineConfig
from video_shot_breakdown.logger import info, error


class LLMPromptConverter(BasePromptConverter, BaseAgent):
    """基于LLM的提示词转换器"""

    def __init__(self, llm_client, config: Optional[HengLineConfig]):
        super().__init__(config)
        self.llm_client = llm_client

    def convert(self, fragment_sequence: FragmentSequence) -> AIVideoInstructions:
        """使用LLM转换提示词"""
        info(f"使用LLM转换提示词，片段数: {len(fragment_sequence.fragments)}")

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
        """使用LLM转换单个片段"""
        # 准备提示词
        user_prompt = self._get_prompt_template("prompt_converter").format(
            fragment_id=fragment.id,
            description=fragment.description,
            duration=fragment.duration,
            character=fragment.continuity_notes.get("main_character", "角色"),
            location=fragment.continuity_notes.get("location", "场景"),
            model=self.config.target_model
        )

        # 调用LLM
        system_prompt = "你是一位顶尖的AI视频转换大师，可以将视频片段的描述转换为适合扩散模型的提示词描述。"

        # 调用LLM
        result = self._call_llm_parse_with_retry(self.llm_client, system_prompt, user_prompt)

        # 创建提示词对象
        return AIVideoPrompt(
            fragment_id=fragment.id,
            prompt=result.get("prompt", fragment.description),
            negative_prompt=result.get("negative_prompt", self.config.default_negative_prompt),
            duration=fragment.duration,
            model=self.config.target_model,
            style=result.get("style_hint")
        )
