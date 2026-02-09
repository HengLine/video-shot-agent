"""
@FileName: llm_video_splitter.py
@Description: 基于LLM的视频智能分割器，保持视频连贯性与一致性
@Author: HengLine
@Github: https://github.com/HengLine/video-shot-agent
@Time: 2026/1/26 22:30
"""
from typing import List, Optional, Dict, Any
import json
import time

from video_shot_breakdown.hengline.agent.base_agent import BaseAgent
from video_shot_breakdown.hengline.agent.shot_segmenter.shot_segmenter_models import ShotSequence, ShotInfo
from video_shot_breakdown.hengline.agent.video_splitter.base_video_splitter import BaseVideoSplitter
from video_shot_breakdown.hengline.agent.video_splitter.rule_video_splitter import RuleVideoSplitter
from video_shot_breakdown.hengline.agent.video_splitter.video_splitter_models import FragmentSequence, VideoFragment
from video_shot_breakdown.hengline.hengline_config import HengLineConfig
from video_shot_breakdown.logger import info, error, warning, debug
from video_shot_breakdown.utils.log_utils import print_log_exception


class LLMVideoSplitter(BaseVideoSplitter, BaseAgent):
    """基于LLM的视频智能分割器 - 保持连贯性与一致性"""

    def __init__(self, llm_client, config: Optional[HengLineConfig]):
        super().__init__(config)
        self.llm_client = llm_client
        self.rule_splitter = RuleVideoSplitter(config)  # 备用规则分割器
        self.split_cache = {}  # 缓存分割决策，避免重复计算

        # 分割阈值配置
        self.split_threshold = getattr(config, 'llm_split_threshold', 5.0)  # 超过5秒触发AI分割
        self.min_split_segment = getattr(config, 'min_fragment_duration', 1.0)  # 最小分割片段
        self.max_split_segment = getattr(config, 'max_fragment_duration', 5.0)  # 最大分割片段

    def cut(self, shot_sequence: ShotSequence) -> FragmentSequence:
        """使用LLM智能分割视频，保持连贯性"""
        info(f"开始智能视频分割，镜头数: {len(shot_sequence.shots)}")

        fragments = []
        current_time = 0.0
        fragment_id_counter = 0

        # 收集场景和角色信息，用于保持一致性
        scene_context = self._collect_scene_context(shot_sequence)

        source_info = {
            "shot_count": len(shot_sequence.shots),
            "original_duration": shot_sequence.stats.get("total_duration", 0.0),
            "split_method": "llm_adaptive",
            "scene_context": scene_context
        }

        for shot_idx, shot in enumerate(shot_sequence.shots):
            try:
                debug(f"处理镜头 {shot.id}: {shot.description} (时长: {shot.duration}s)")

                # 判断是否需要AI分割
                if self._should_use_llm_split(shot):
                    info(f"镜头 {shot.id} 时长({shot.duration}s)超过阈值，使用AI分割")

                    # 准备上下文信息
                    context = {
                        "shot": shot,
                        "prev_shot": shot_sequence.shots[shot_idx-1] if shot_idx > 0 else None,
                        "next_shot": shot_sequence.shots[shot_idx+1] if shot_idx < len(shot_sequence.shots)-1 else None,
                        "scene_context": scene_context,
                        "current_time": current_time,
                        "fragment_offset": fragment_id_counter
                    }

                    # 使用AI分割
                    shot_fragments = self._split_shot_with_llm(context)

                    # 验证分割结果
                    validated_fragments = self._validate_and_adjust_fragments(
                        shot_fragments, shot, current_time, fragment_id_counter
                    )

                    fragments.extend(validated_fragments)
                    fragment_id_counter += len(validated_fragments)

                    if validated_fragments:
                        current_time = validated_fragments[-1].start_time + validated_fragments[-1].duration

                else:
                    # 使用规则分割器
                    debug(f"镜头 {shot.id} 使用规则分割")
                    rule_fragments = self.rule_splitter.split_shot(
                        shot, current_time, fragment_id_counter
                    )

                    fragments.extend(rule_fragments)
                    fragment_id_counter += len(rule_fragments)

                    if rule_fragments:
                        current_time = rule_fragments[-1].start_time + rule_fragments[-1].duration

            except Exception as e:
                error(f"镜头{shot.id}分割失败: {str(e)}")
                print_log_exception()
                warning(f"镜头{shot.id}降级到简单规则分割")

                # 紧急降级
                fallback_fragments = self.rule_splitter.split_shot(
                    shot, current_time, fragment_id_counter
                )
                fragments.extend(fallback_fragments)
                fragment_id_counter += len(fallback_fragments)

                if fallback_fragments:
                    current_time = fallback_fragments[-1].start_time + fallback_fragments[-1].duration

        # 创建片段序列
        fragment_sequence = FragmentSequence(
            source_info=source_info,
            fragments=fragments,
            metadata={
                "split_method": "llm_adaptive",
                "ai_split_count": sum(1 for f in fragments if f.metadata.get("split_by", "") == "ai"),
                "rule_split_count": sum(1 for f in fragments if f.metadata.get("split_by", "") == "rule"),
                "total_fragments": len(fragments),
                "average_duration": sum(f.duration for f in fragments) / len(fragments) if fragments else 0
            }
        )

        info(f"视频分割完成: 共生成{len(fragments)}个片段")
        return self.post_process(fragment_sequence)

    def _should_use_llm_split(self, shot: ShotInfo) -> bool:
        """判断是否应该使用AI分割"""
        # 基础条件：时长超过阈值
        if shot.duration <= self.split_threshold:
            return False

        # 复杂镜头类型更适合AI分割
        complex_shots = ["ACTION", "MOVING", "PANORAMA", "ZOOM"]
        if shot.shot_type in complex_shots:
            return True

        # 长对话场景
        if shot.description and any(keyword in shot.description.lower() for keyword in ["对话", "交谈", "讨论", "talk", "conversation"]):
            return True

        return True  # 默认超过阈值就用AI

    def _collect_scene_context(self, shot_sequence: ShotSequence) -> Dict[str, Any]:
        """收集场景上下文信息"""
        scene_context = {
            "scenes": {},
            "characters": {},
            "locations": {},
            "mood": {}
        }

        for shot in shot_sequence.shots:
            # 按场景分组
            scene_id = shot.scene_id
            if scene_id not in scene_context["scenes"]:
                scene_context["scenes"][scene_id] = {
                    "shot_ids": [],
                    "duration": 0,
                    "main_characters": set(),
                    "mood": shot.mood if hasattr(shot, 'mood') else "neutral"
                }

            scene_context["scenes"][scene_id]["shot_ids"].append(shot.id)
            scene_context["scenes"][scene_id]["duration"] += shot.duration

            if shot.main_character:
                characters = shot.main_character.split(",")
                for char in characters:
                    char = char.strip()
                    scene_context["scenes"][scene_id]["main_characters"].add(char)

                    # 记录角色出现
                    if char not in scene_context["characters"]:
                        scene_context["characters"][char] = {
                            "scenes": set(),
                            "total_duration": 0
                        }
                    scene_context["characters"][char]["scenes"].add(scene_id)
                    scene_context["characters"][char]["total_duration"] += shot.duration

        return scene_context

    def _split_shot_with_llm(self, context: Dict[str, Any]) -> List[VideoFragment]:
        """使用LLM智能分割单个镜头，保持连贯性"""
        shot = context["shot"]
        cache_key = f"{shot.id}_{shot.duration}_{hash(shot.description)}"

        # 检查缓存
        if cache_key in self.split_cache:
            debug(f"使用缓存的分割决策: {shot.id}")
            return self._create_fragments_from_cache(
                self.split_cache[cache_key], context
            )

        # 准备上下文提示词
        user_prompt = self._get_enhanced_prompt_template(context)
        system_prompt = self._get_system_prompt()

        debug(f"调用LLM分割镜头 {shot.id}")
        start_time = time.time()

        try:
            # 调用LLM
            llm_response = self._call_llm_parse_with_retry(
                self.llm_client, system_prompt, user_prompt
            )

            response_time = time.time() - start_time
            debug(f"LLM响应时间: {response_time:.2f}s")

            # 解析响应
            if isinstance(llm_response, str):
                try:
                    decision = json.loads(llm_response)
                except json.JSONDecodeError:
                    error(f"LLM返回非JSON格式: {llm_response[:100]}...")
                    raise ValueError("LLM返回格式错误")
            else:
                decision = llm_response

            # 验证决策
            self._validate_llm_decision(decision, shot)

            # 缓存决策
            self.split_cache[cache_key] = decision

            # 创建片段
            fragments = self._create_fragments_from_decision(decision, context)

            return fragments

        except Exception as e:
            error(f"LLM分割失败: {str(e)}")
            raise

    def _get_enhanced_prompt_template(self, context: Dict[str, Any]) -> str:
        """获取增强的提示词模板"""
        shot = context["shot"]
        prev_shot = context["prev_shot"]
        next_shot = context["next_shot"]
        scene_context = context["scene_context"]

        # 构建详细上下文
        scene_info = ""
        if shot.scene_id in scene_context["scenes"]:
            scene_data = scene_context["scenes"][shot.scene_id]
            characters = ", ".join(scene_data["main_characters"])
            scene_info = f"场景{shot.scene_id}: 时长{scene_data['duration']}秒, 角色[{characters}], 氛围{scene_data.get('mood', '中性')}"

        prev_context = ""
        if prev_shot:
            prev_context = f"前一个镜头: {prev_shot.description} ({prev_shot.duration}s, {prev_shot.shot_type})"

        next_context = ""
        if next_shot:
            next_context = f"后一个镜头: {next_shot.description} ({next_shot.duration}s, {next_shot.shot_type})"

        prompt_template = self._get_prompt_template("video_splitter")

        return prompt_template.format(
            shot_id=shot.id,
            description=shot.description,
            duration=shot.duration,
            shot_type=shot.shot_type.value if hasattr(shot.shot_type, 'value') else shot.shot_type,
            main_character=shot.main_character or "无",
            scene_info=scene_info,
            prev_context=prev_context,
            next_context=next_context,
            split_threshold=self.split_threshold,
            min_segment=self.min_split_segment,
            max_segment=self.max_split_segment,
            continuity_notes=self._get_continuity_notes(shot, context)
        )

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一位顶尖的电影剪辑师和分镜设计师，精通视觉叙事和节奏控制。
            你的任务是将较长的视频镜头智能地分割为适合AI视频生成的片段（每个片段通常2-5秒）。
            
            请遵循以下原则：
            1. 保持连贯性：分割点应该选择在动作自然停顿、对话间隙、场景转换处
            2. 保持一致性：确保角色外观、场景布置、灯光风格在分割后保持一致
            3. 叙事流畅：分割后的片段应该能连贯地讲述原镜头的故事
            4. 节奏控制：根据内容调整片段时长，动作场景可以短一些，对话场景可以长一些
            
            请基于镜头内容和前后文关系，给出最合理的分割方案。"""

    def _get_continuity_notes(self, shot: ShotInfo, context: Dict) -> str:
        """生成连续性说明"""
        notes = []

        # 角色连续性
        if shot.main_character:
            notes.append(f"主要角色: {shot.main_character}")

        # 场景连续性
        if hasattr(shot, 'scene_id'):
            notes.append(f"场景ID: {shot.scene_id}")

        # 视觉元素连续性
        if hasattr(shot, 'visual_elements'):
            elements = shot.visual_elements.split(",") if shot.visual_elements else []
            if elements:
                notes.append(f"关键视觉元素: {', '.join(elements[:3])}")

        # 动作连续性
        if "动作" in shot.description or "move" in shot.description.lower():
            notes.append("注意动作的连贯衔接")

        return "; ".join(notes) if notes else "无特殊连续性要求"

    def _validate_llm_decision(self, decision: Dict, shot: ShotInfo) -> None:
        """验证LLM决策的合理性"""
        if not isinstance(decision, dict):
            raise ValueError(f"决策必须是字典格式，实际是{type(decision)}")

        if "needs_split" not in decision:
            raise ValueError("决策中缺少 needs_split 字段")

        if decision.get("needs_split", False):
            if "segments" not in decision:
                raise ValueError("需要分割但缺少 segments 字段")

            segments = decision["segments"]
            if not isinstance(segments, list) or len(segments) == 0:
                raise ValueError("segments 必须是非空列表")

            # 检查每个片段
            total_duration = 0
            for i, segment in enumerate(segments):
                if "duration" not in segment:
                    raise ValueError(f"片段{i+1}缺少duration字段")

                duration = segment["duration"]
                if not isinstance(duration, (int, float)) or duration <= 0:
                    raise ValueError(f"片段{i+1}的duration必须是正数")

                if duration < self.min_split_segment:
                    raise ValueError(f"片段{i+1}时长({duration}s)低于最小值({self.min_split_segment}s)")

                if duration > self.max_split_segment:
                    raise ValueError(f"片段{i+1}时长({duration}s)超过最大值({self.max_split_segment}s)")

                total_duration += duration

            # 检查总时长匹配
            if abs(total_duration - shot.duration) > 1.0:  # 允许1秒误差
                warning(f"分割总时长({total_duration}s)与镜头时长({shot.duration}s)不匹配")

    def _create_fragments_from_decision(self, decision: Dict, context: Dict) -> List[VideoFragment]:
        """根据LLM决策创建片段"""
        shot = context["shot"]
        current_time = context["current_time"]
        fragment_offset = context["fragment_offset"]

        fragments = []

        if decision.get("needs_split", False):
            segments = decision["segments"]

            for seg_idx, segment in enumerate(segments):
                fragment_id = f"frag_{fragment_offset + len(fragments) + 1:03d}_s{seg_idx+1}"

                # 计算开始时间
                segment_start_time = current_time + sum(
                    s.get("duration", 0) for s in segments[:seg_idx]
                )

                fragment = VideoFragment(
                    id=fragment_id,
                    shot_id=shot.id,
                    element_ids=shot.element_ids if seg_idx == 0 else [],
                    start_time=round(segment_start_time, 2),
                    duration=segment["duration"],
                    description=segment.get("description", f"{shot.description} - 部分{seg_idx+1}"),
                    continuity_notes={
                        "main_character": shot.main_character,
                        "location": f"场景{shot.scene_id}",
                        "continuity_id": f"{shot.id}_seq{seg_idx+1}",
                        "prev_fragment": fragments[-1].id if fragments else None,
                        "split_reason": decision.get("reason", "AI智能分割")
                    },
                    metadata={
                        "split_by": "ai",
                        "original_shot": shot.id,
                        "segment_index": seg_idx,
                        "total_segments": len(segments),
                        "ai_decision": decision.get("reason", ""),
                        "timestamp": time.time()
                    }
                )
                fragments.append(fragment)
        else:
            # 不分割
            fragment_id = f"frag_{fragment_offset + 1:03d}"
            fragment = VideoFragment(
                id=fragment_id,
                shot_id=shot.id,
                element_ids=shot.element_ids,
                start_time=current_time,
                duration=min(shot.duration, self.max_split_segment),
                description=shot.description,
                continuity_notes={
                    "main_character": shot.main_character,
                    "location": f"场景{shot.scene_id}",
                    "continuity_id": f"{shot.id}_whole",
                    "split_reason": decision.get("reason", "无需分割")
                },
                metadata={
                    "split_by": "ai",
                    "original_shot": shot.id,
                    "segment_index": 0,
                    "total_segments": 1,
                    "ai_decision": decision.get("reason", ""),
                    "timestamp": time.time()
                }
            )
            fragments.append(fragment)

        return fragments

    def _create_fragments_from_cache(self, decision: Dict, context: Dict) -> List[VideoFragment]:
        """从缓存创建片段"""
        # 创建副本避免修改缓存
        cached_decision = decision.copy()
        if "segments" in cached_decision:
            cached_decision["segments"] = [seg.copy() for seg in cached_decision["segments"]]

        return self._create_fragments_from_decision(cached_decision, context)

    def _validate_and_adjust_fragments(self, fragments: List[VideoFragment],
                                      shot: ShotInfo, current_time: float,
                                      fragment_offset: int) -> List[VideoFragment]:
        """验证并调整分割片段"""
        if not fragments:
            warning(f"镜头{shot.id}分割结果为空，使用规则分割")
            return self.rule_splitter.split_shot(shot, current_time, fragment_offset)

        # 验证总时长
        total_duration = sum(f.duration for f in fragments)
        if abs(total_duration - shot.duration) > 2.0:  # 允许2秒误差
            warning(f"镜头{shot.id}分割总时长({total_duration}s)与原始时长({shot.duration}s)差异过大")
            # 重新分配时长
            scale_factor = shot.duration / total_duration
            for fragment in fragments:
                fragment.duration = round(fragment.duration * scale_factor, 2)

        # 确保连续性
        prev_end_time = current_time
        for i, fragment in enumerate(fragments):
            fragment.start_time = prev_end_time
            prev_end_time += fragment.duration

            # 更新连续性ID
            if i > 0:
                fragment.continuity_notes["prev_fragment"] = fragments[i-1].id

        return fragments