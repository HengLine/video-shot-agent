"""
@FileName: llm_video_splitter.py
@Description: 
@Author: HengLine
@Time: 2026/1/26 22:30
"""
from typing import Dict, Any, List, Optional

from hengline.agent.base_agent import BaseAgent
from hengline.agent.shot_segmenter.shot_segmenter_models import ShotSequence, ShotInfo
from hengline.agent.video_splitter.base_video_splitter import BaseVideoSplitter
from hengline.agent.video_splitter.rule_video_splitter import RuleVideoSplitter
from hengline.agent.video_splitter.video_splitter_models import FragmentSequence, VideoFragment
from hengline.logger import info, error
from utils.log_utils import print_log_exception


class LLMVideoSplitter(BaseVideoSplitter, BaseAgent):
    """基于LLM的视频分割器（备用）"""

    def __init__(self, llm_client, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.llm_client = llm_client

    def cut(self, shot_sequence: ShotSequence) -> FragmentSequence:
        """使用LLM决定如何分割"""
        info(f"使用LLM分割视频，镜头数: {len(shot_sequence.shots)}")

        fragments = []
        current_time = 0.0
        source_info = {
            "shot_count": len(shot_sequence.shots),
            "original_duration": shot_sequence.stats.get("total_duration", 0.0)
        }

        for shot in shot_sequence.shots:
            try:
                shot_fragments = self._split_shot_with_llm(shot, current_time, len(fragments))
                fragments.extend(shot_fragments)

                if shot_fragments:
                    current_time = shot_fragments[-1].start_time + shot_fragments[-1].duration

            except Exception as e:
                error(f"镜头{shot.id}分割失败: {str(e)}")
                print_log_exception()
                # 降级到简单规则
                simple_cutter = RuleVideoSplitter()
                fallback_fragments = simple_cutter.split_shot(shot, current_time, len(fragments))
                fragments.extend(fallback_fragments)

        fragment_sequence = FragmentSequence(
            source_info=source_info,
            fragments=fragments
        )
        return self.post_process(fragment_sequence)

    def _split_shot_with_llm(self, shot: ShotInfo, start_time: float, fragment_offset: int) -> List[VideoFragment]:
        """使用LLM分割单个镜头"""
        # 准备提示词
        user_prompt = self._get_prompt_template("video_splitter").format(
            shot_id=shot.id,
            description=shot.description,
            duration=shot.duration,
            shot_type=shot.shot_type.value,
            main_character=shot.main_character or "无"
        )

        system_prompt = "你是一位顶尖的视频分割大师，精通分镜设计和视觉叙事，能将不同的镜头根据时间切割为符合范围的片段。"

        # 调用LLM
        decision = self._call_llm_parse_with_retry(self.llm_client, system_prompt, user_prompt)

        # 根据LLM决策创建片段
        fragments = []

        if decision.get("needs_split", False):
            segments = decision.get("segments", [])
            for seg_idx, segment in enumerate(segments):
                fragment_id = f"frag_{fragment_offset + len(fragments) + 1:03d}_{seg_idx + 1}"

                fragment = VideoFragment(
                    id=fragment_id,
                    shot_id=shot.id,
                    element_ids=shot.element_ids if seg_idx == 0 else [],
                    start_time=start_time + sum(s["duration"] for s in segments[:seg_idx]),
                    duration=segment.get("duration", 2.5),
                    description=segment.get("description", f"{shot.description} 部分{seg_idx + 1}"),
                    continuity_notes={
                        "main_character": shot.main_character,
                        "location": f"场景{shot.scene_id}"
                    }
                )
                fragments.append(fragment)
        else:
            # 不分割，直接作为一个片段
            fragment_id = self._generate_fragment_id(fragment_offset)
            fragment = VideoFragment(
                id=fragment_id,
                shot_id=shot.id,
                element_ids=shot.element_ids,
                start_time=start_time,
                duration=shot.duration,
                description=shot.description,
                continuity_notes={
                    "main_character": shot.main_character,
                    "location": f"场景{shot.scene_id}"
                }
            )
            fragments.append(fragment)

        return fragments
