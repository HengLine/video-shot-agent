"""
@FileName: duration_segmenter.py
@Description: 5秒分片算法
@Author: HengLine
@Time: 2025/12/20 21:26
"""
from typing import Dict, List, Any, Tuple

from hengline.logger import info, debug, warning
from hengline.config.temporal_planner_config import SegmentConfig
from ..temporal_planner_model import TimeSegment, TimelineEvent, DurationEstimation
from ...script_parser.script_parser_model import UnifiedScript


class DurationSegmenter:
    """五秒分段器，将视频按5秒段落进行智能分段"""

    def __init__(self, config: SegmentConfig):
        """
        初始化五秒分段器

        参数:
            config: 配置字典
                - target_duration: 目标分段时长（默认5秒）
                - min_duration: 最小分段时长（默认2秒）
                - max_duration: 最大分段时长（默认8秒）
                - optimize_boundaries: 是否优化边界（默认True）
                - split_long_segments: 是否分割长段落（默认True）
        """
        self.config = config

    def segment_timeline(self, script_input: UnifiedScript,
                         duration_estimations: Dict[str, DurationEstimation]) -> List[TimeSegment]:
        """
        将脚本输入和时间估算分割为5秒段落

        参数:
            script_input: 统一脚本输入
            duration_estimations: 元素时长估算字典

        返回:
            时间分段列表
        """
        info("开始分段处理")

        # 1. 将脚本输入转换为时间线事件
        timeline_events = self._convert_to_timeline_events(script_input, duration_estimations)

        if not timeline_events:
            warning("没有生成时间线事件")
            return []

        # 2. 计算总时长
        total_duration = self._calculate_total_duration(timeline_events)
        info(f"总时长: {total_duration:.1f}秒，事件数: {len(timeline_events)}")

        # 3. 按场景分组
        scenes = self._group_by_scene(timeline_events)
        info(f"识别到 {len(scenes)} 个场景")

        # 4. 为每个场景生成初始分段
        all_segments = []
        for scene_idx, scene_events in enumerate(scenes):
            scene_segments = self._segment_scene(scene_events, scene_idx)
            all_segments.extend(scene_segments)

        info(f"生成初始分段: {len(all_segments)} 个")

        # 5. 优化分段边界
        if self.config.optimize_boundaries:
            all_segments = self._optimize_segment_boundaries(all_segments, timeline_events)
            info(f"边界优化后: {len(all_segments)} 个分段")

        # 6. 分割长段落
        if self.config.split_long_segments:
            all_segments = self._split_long_segments(all_segments, timeline_events)
            info(f"分割长段落后: {len(all_segments)} 个分段")

        # 7. 确保分段时长在合理范围内
        all_segments = self._validate_segment_durations(all_segments)

        # 8. 添加连续性锚点
        all_segments = self._add_continuity_anchors(all_segments)

        # 9. 分析分段质量
        quality_analysis = self.analyze_segmentation_quality(all_segments)
        info(f"分段质量分析: {quality_analysis['summary']}")

        info(f"分段处理完成，最终分段数: {len(all_segments)}")
        return all_segments

    def _convert_to_timeline_events(self, script_input: UnifiedScript,
                                    duration_estimations: Dict[str, DurationEstimation]) -> List[TimelineEvent]:
        """
        将脚本输入转换为时间线事件

        参数:
            script_input: 统一脚本输入
            duration_estimations: 元素时长估算字典

        返回:
            时间线事件列表
        """
        timeline_events = []
        current_time = 0.0

        if not script_input:
            warning("脚本输入为空，无法转换为时间线事件")
            return timeline_events

        # 获取所有元素
        all_elements = script_input.get_all_elements()

        # 按类型和位置排序元素
        sorted_elements = self._sort_elements(all_elements)

        # 处理场景信息
        scene_mapping = {}
        for scene in script_input.scenes:
            if 'id' in scene:
                scene_mapping[scene.scene_id] = scene

        # 转换每个元素为时间线事件
        for element in sorted_elements:
            element_id = element.get('id', '')
            element_type = element.get('type', '')

            # 获取时长估算
            duration_est = duration_estimations.get(element_id)
            if not duration_est:
                warning(f"元素 {element_id} 没有时长估算，使用默认时长")
                duration = self._get_default_duration(element_type)
                confidence = 0.5
            else:
                duration = duration_est.estimated_duration
                confidence = duration_est.confidence

            # 获取场景信息
            scene_id = element.get('scene_id', '')
            scene_info = scene_mapping.get(scene_id, {})

            # 创建时间线事件
            event = TimelineEvent(
                type=element_type,
                element_id=element_id,
                scene_id=scene_id,
                start_time=current_time,
                duration=duration,
                content=element.get('content', '') or element.get('text', ''),
                speaker=element.get('speaker') or element.get('character'),
                actor=element.get('actor') or element.get('character'),
                emotion=element.get('emotion'),
                intensity=element.get('intensity')
            )

            timeline_events.append(event)

            # 更新时间
            current_time += duration

            # 如果这是场景的第一个元素，添加场景开始事件
            if scene_id and scene_id not in [e.scene_id for e in timeline_events if e.type == 'scene_start']:
                scene_start_event = TimelineEvent(
                    type='scene_start',
                    element_id=f'scene_start_{scene_id}',
                    scene_id=scene_id,
                    start_time=current_time - duration,  # 场景开始时间与第一个元素相同
                    duration=0.1,  # 场景开始事件持续很短时间
                    content=scene_info.get('description', f'场景 {scene_id}'),
                    speaker=None,
                    actor=None,
                    emotion=scene_info.get('mood'),
                    intensity=None
                )
                # 插入到当前事件之前
                timeline_events.insert(len(timeline_events) - 1, scene_start_event)

        # 按开始时间排序
        timeline_events.sort(key=lambda x: x.start_time)

        info(f"转换了 {len(timeline_events)} 个时间线事件")
        return timeline_events

    def _sort_elements(self, elements: List[Dict]) -> List[Dict]:
        """
        对元素进行排序

        参数:
            elements: 元素列表

        返回:
            排序后的元素列表
        """
        # 按场景分组
        scenes = {}
        for element in elements:
            scene_id = element.get('scene_id', 'default')
            if scene_id not in scenes:
                scenes[scene_id] = []
            scenes[scene_id].append(element)

        # 在每个场景内按顺序排序
        sorted_elements = []
        for scene_id in sorted(scenes.keys()):
            scene_elements = scenes[scene_id]

            # 先按类型排序：描述 -> 动作 -> 对话
            type_order = {'description': 0, 'action': 1, 'dialogue': 2}
            scene_elements.sort(key=lambda x: type_order.get(x.get('type', ''), 3))

            # 再按位置或序号排序
            scene_elements.sort(key=lambda x: x.get('order', 0) or x.get('position', 0))

            sorted_elements.extend(scene_elements)

        return sorted_elements

    def _get_default_duration(self, element_type: str) -> float:
        """
        获取默认时长

        参数:
            element_type: 元素类型

        返回:
            默认时长（秒）
        """
        default_durations = {
            'dialogue': 3.0,  # 平均对话时长
            'action': 2.5,  # 平均动作时长
            'description': 2.0,  # 平均描述时长
            'scene_start': 0.1  # 场景开始事件
        }
        return default_durations.get(element_type, 2.0)

    def _calculate_total_duration(self, timeline_events: List['TimelineEvent']) -> float:
        """
        计算总时长

        参数:
            timeline_events: 时间线事件列表

        返回:
            总时长（秒）
        """
        if not timeline_events:
            return 0.0

        # 找到最后一个事件的结束时间
        max_end_time = 0.0
        for event in timeline_events:
            event_end = event.start_time + event.duration
            if event_end > max_end_time:
                max_end_time = event_end

        return max_end_time

    def _group_by_scene(self, timeline_events: List['TimelineEvent']) -> List[List['TimelineEvent']]:
        """
        根据场景转换点将时间线事件分组为场景

        参数:
            timeline_events: 时间线事件列表

        返回:
            按场景分组的事件列表
        """
        if not timeline_events:
            return []

        # 按开始时间排序
        sorted_events = sorted(timeline_events, key=lambda x: x.start_time)

        scenes = []
        current_scene = []
        current_scene_id = ""

        for event in sorted_events:
            # 检查是否是新的场景开始
            is_scene_start = event.type == "scene_start"
            has_new_scene_id = event.scene_id and event.scene_id != current_scene_id

            if is_scene_start or has_new_scene_id:
                # 保存当前场景（如果有内容）
                if current_scene:
                    scenes.append(current_scene.copy())
                    current_scene = []

                current_scene_id = event.scene_id if event.scene_id else f"scene_{len(scenes) + 1}"

            current_scene.append(event)

        # 添加最后一个场景
        if current_scene:
            scenes.append(current_scene)

        debug(f"将时间线分为 {len(scenes)} 个场景")
        return scenes

    def _segment_scene(self, scene_events: List['TimelineEvent'],
                       scene_idx: int) -> List[TimeSegment]:
        """
        为单个场景生成初始分段

        参数:
            scene_events: 场景事件列表
            scene_idx: 场景索引

        返回:
            场景分段列表
        """
        if not scene_events:
            return []

        # 获取场景时间范围
        start_time = min(e.start_time for e in scene_events)
        end_time = max(e.start_time + e.duration for e in scene_events)
        scene_duration = end_time - start_time

        # 计算建议的分段数
        target_duration = self.config.target_segment_duration
        suggested_segments = max(1, int(round(scene_duration / target_duration)))

        # 基于事件密度调整分段数
        event_density = len(scene_events) / scene_duration
        if event_density > 1.5:  # 高密度事件
            suggested_segments = max(suggested_segments, int(scene_duration / 4))
        elif event_density < 0.3:  # 低密度事件
            suggested_segments = max(1, int(scene_duration / 6))

        # 限制最大分段数
        suggested_segments = min(suggested_segments, self.config.max_segments_per_scene)

        # 生成分段时间点
        segment_duration = scene_duration / suggested_segments
        segments = []

        for i in range(suggested_segments):
            seg_start = start_time + i * segment_duration
            seg_end = start_time + (i + 1) * segment_duration

            # 调整最后一个分段以精确结束
            if i == suggested_segments - 1:
                seg_end = end_time

            # 获取分段内的事件
            seg_events = [
                e for e in scene_events
                if e.start_time >= seg_start and e.start_time < seg_end
            ]

            # 创建分段
            segment = self._create_segment(
                scene_idx=scene_idx,
                segment_idx=i,
                time_range=(seg_start, seg_end),
                events=seg_events,
                all_events=scene_events
            )

            segments.append(segment)

        debug(f"场景 {scene_idx}: 生成 {len(segments)} 个初始分段")
        return segments

    def _create_segment(self, scene_idx: int, segment_idx: int,
                        time_range: Tuple[float, float],
                        events: List['TimelineEvent'],
                        all_events: List['TimelineEvent']) -> TimeSegment:
        """
        创建单个时间分段

        参数:
            scene_idx: 场景索引
            segment_idx: 分段索引
            time_range: 时间范围 (开始, 结束)
            events: 分段内的事件
            all_events: 所有事件（用于上下文）

        返回:
            时间分段对象
        """
        start_time, end_time = time_range
        duration = end_time - start_time

        # 提取内容
        visual_content = self._extract_visual_content(events, start_time, end_time)
        audio_content = self._extract_audio_content(events, start_time, end_time)
        key_elements = self._extract_key_elements(events, start_time, end_time)
        pacing = self._calculate_pacing(events, start_time, end_time)
        continuity_hooks = self._create_continuity_hooks(events, start_time, end_time)

        # 计算质量分数
        quality_score = self._calculate_quality_score(events, duration)

        return TimeSegment(
            segment_id=f"scene_{scene_idx}_seg_{segment_idx}",
            time_range=time_range,
            duration=duration,
            visual_content=visual_content,
            audio_content=audio_content,
            events=[e.element_id for e in events if e.element_id],
            key_elements=key_elements,
            continuity_hooks=continuity_hooks,
            pacing=pacing,
            quality_score=quality_score
        )

    def _extract_visual_content(self, events: List['TimelineEvent'],
                                start_time: float, end_time: float) -> str:
        """
        从事件中提取视觉内容描述

        参数:
            events: 事件列表
            start_time: 开始时间
            end_time: 结束时间

        返回:
            视觉内容描述
        """
        if not events:
            return "待生成视觉内容"

        # 按类型组织事件
        scene_starts = [e for e in events if e.type == "scene_start"]
        actions = [e for e in events if e.type == "action"]
        descriptions = [e for e in events if e.type == "description"]
        dialogues = [e for e in events if e.type == "dialogue"]

        visual_parts = []

        # 场景信息
        if scene_starts:
            scene_desc = scene_starts[0].content
            visual_parts.append(f"场景: {scene_desc[:50]}{'...' if len(scene_desc) > 50 else ''}")

        # 动作描述
        if actions:
            action_texts = []
            for action in actions[:2]:  # 最多2个主要动作
                intensity_suffix = f" (强度{action.intensity})" if action.intensity else ""
                action_texts.append(f"{action.content[:30]}{intensity_suffix}")

            if action_texts:
                visual_parts.append(f"动作: {'; '.join(action_texts)}")

        # 人物信息
        speakers = set(e.speaker for e in dialogues if e.speaker)
        actors = set(e.actor for e in actions if e.actor)
        characters = speakers.union(actors)

        if characters:
            visual_parts.append(f"人物: {', '.join(characters)}")

        # 描述性内容
        if descriptions:
            desc_text = descriptions[0].content[:60]
            if len(descriptions) > 1 or len(descriptions[0].content) > 60:
                desc_text += "..."
            visual_parts.append(f"描述: {desc_text}")

        # 情绪信息
        emotions = set(e.emotion for e in events if e.emotion)
        if emotions:
            visual_parts.append(f"情绪: {', '.join(emotions)}")

        return " | ".join(visual_parts) if visual_parts else "常规视觉内容"

    def _extract_audio_content(self, events: List['TimelineEvent'],
                               start_time: float, end_time: float) -> str:
        """
        从事件中提取音频内容描述

        参数:
            events: 事件列表
            start_time: 开始时间
            end_time: 结束时间

        返回:
            音频内容描述
        """
        if not events:
            return "待生成音频内容"

        audio_parts = []
        duration = end_time - start_time

        # 对话分析
        dialogues = [e for e in events if e.type == "dialogue"]
        if dialogues:
            total_dialogue_duration = sum(d.duration for d in dialogues)
            dialogue_ratio = total_dialogue_duration / duration if duration > 0 else 0

            if dialogue_ratio > 0.7:
                audio_parts.append("密集对话")
            elif dialogue_ratio > 0.4:
                audio_parts.append("主要对话")
            else:
                audio_parts.append("部分对话")

            # 说话者信息
            speakers = set(d.speaker for d in dialogues if d.speaker)
            if speakers:
                audio_parts.append(f"{len(speakers)}位说话者")

        # 动作音效
        actions = [e for e in events if e.type == "action"]
        if actions:
            intense_actions = [a for a in actions if a.intensity and a.intensity >= 7]
            if intense_actions:
                audio_parts.append("强烈音效")
            else:
                audio_parts.append("动作音效")

        # 环境音
        scene_starts = [e for e in events if e.type == "scene_start"]
        if scene_starts:
            scene_desc = scene_starts[0].content.lower()
            if any(word in scene_desc for word in ["室内", "房间", "办公室"]):
                audio_parts.append("室内环境音")
            elif any(word in scene_desc for word in ["室外", "街道", "森林"]):
                audio_parts.append("室外环境音")

        # 情绪氛围
        emotions = set(e.emotion for e in events if e.emotion)
        if emotions:
            if any(e in emotions for e in ["紧张", "激烈", "恐怖"]):
                audio_parts.append("紧张氛围")
            elif any(e in emotions for e in ["平静", "轻松", "温馨"]):
                audio_parts.append("舒缓氛围")

        return " | ".join(audio_parts) if audio_parts else "常规音频内容"

    def _extract_key_elements(self, events: List['TimelineEvent'],
                              start_time: float, end_time: float) -> List[str]:
        """
        提取关键元素

        参数:
            events: 事件列表
            start_time: 开始时间
            end_time: 结束时间

        返回:
            关键元素列表
        """
        if not events:
            return []

        key_elements = set()

        # 人物元素
        speakers = set(e.speaker for e in events if e.speaker)
        actors = set(e.actor for e in events if e.actor)

        for character in speakers.union(actors):
            key_elements.add(f"人物:{character}")

        # 重要动作
        intense_actions = [e for e in events
                           if e.type == "action" and e.intensity and e.intensity >= 7]
        if intense_actions:
            key_elements.add("强烈动作")

        # 场景类型
        scene_starts = [e for e in events if e.type == "scene_start"]
        if scene_starts:
            scene_desc = scene_starts[0].content.lower()
            scene_types = []

            if any(word in scene_desc for word in ["室内", "房间", "屋内"]):
                scene_types.append("室内")
            if any(word in scene_desc for word in ["室外", "户外", "外面"]):
                scene_types.append("室外")
            if any(word in scene_desc for word in ["夜景", "夜晚", "黑夜"]):
                scene_types.append("夜景")
            if any(word in scene_desc for word in ["日景", "白天", "早晨"]):
                scene_types.append("日景")

            for scene_type in scene_types:
                key_elements.add(f"场景:{scene_type}")

        # 情绪元素
        emotions = set(e.emotion for e in events if e.emotion)
        for emotion in emotions:
            key_elements.add(f"情绪:{emotion}")

        # 对话特征
        if speakers:
            key_elements.add(f"对话:{len(speakers)}人")

        return list(key_elements)[:5]  # 最多5个关键元素

    def _calculate_pacing(self, events: List['TimelineEvent'],
                          start_time: float, end_time: float) -> str:
        """
        计算节奏

        参数:
            events: 事件列表
            start_time: 开始时间
            end_time: 结束时间

        返回:
            节奏描述
        """
        if not events:
            return "normal"

        duration = end_time - start_time

        # 事件密度
        event_density = len(events) / duration

        # 事件间隔分析
        if len(events) > 1:
            sorted_times = sorted([e.start_time for e in events])
            intervals = [sorted_times[i + 1] - sorted_times[i] for i in range(len(sorted_times) - 1)]
            avg_interval = sum(intervals) / len(intervals) if intervals else duration
        else:
            avg_interval = duration

        # 动作强度
        actions = [e for e in events if e.type == "action"]
        avg_intensity = 0
        if actions:
            intensities = [e.intensity for e in actions if e.intensity]
            if intensities:
                avg_intensity = sum(intensities) / len(intensities)

        # 判断节奏
        if event_density > 1.2 or avg_interval < 0.8 or avg_intensity > 6:
            return "fast"
        elif event_density < 0.4 or avg_interval > 2.5 or (avg_intensity > 0 and avg_intensity < 3):
            return "slow"
        elif event_density > 0.8 and len(set(e.type for e in events)) > 2:
            return "varying"
        else:
            return "normal"

    def _create_continuity_hooks(self, events: List['TimelineEvent'],
                                 start_time: float, end_time: float) -> Dict[str, Any]:
        """
        创建连续性钩子

        参数:
            events: 事件列表
            start_time: 开始时间
            end_time: 结束时间

        返回:
            连续性钩子字典
        """
        hooks = {
            'visual_continuity': [],
            'audio_continuity': [],
            'narrative_elements': [],
            'transition_suggestions': []
        }

        if not events:
            return hooks

        # 视觉连续性
        characters = set()
        for event in events:
            if event.speaker:
                characters.add(event.speaker)
            if event.actor:
                characters.add(event.actor)

        if characters:
            hooks['visual_continuity'].append({
                'type': 'character_presence',
                'characters': list(characters),
                'count': len(characters)
            })

        # 场景连续性
        scene_starts = [e for e in events if e.type == "scene_start"]
        if scene_starts:
            hooks['visual_continuity'].append({
                'type': 'scene_setting',
                'description': scene_starts[0].content[:40]
            })

        # 音频连续性
        dialogues = [e for e in events if e.type == "dialogue"]
        if dialogues:
            hooks['audio_continuity'].append({
                'type': 'speech_present',
                'speaker_count': len(set(d.speaker for d in dialogues if d.speaker)),
                'total_duration': sum(d.duration for d in dialogues)
            })

        # 叙事元素
        key_elements = self._extract_key_elements(events, start_time, end_time)
        if key_elements:
            hooks['narrative_elements'].append({
                'type': 'key_elements',
                'elements': key_elements[:3]
            })

        # 转场建议
        pacing = self._calculate_pacing(events, start_time, end_time)

        if pacing == "fast":
            hooks['transition_suggestions'].append({
                'type': 'quick_cut',
                'reason': 'fast_pacing',
                'suitability': 0.8
            })
        elif pacing == "slow":
            hooks['transition_suggestions'].append({
                'type': 'fade',
                'duration': 0.6,
                'reason': 'slow_pacing',
                'suitability': 0.7
            })
        elif pacing == "varying":
            hooks['transition_suggestions'].append({
                'type': 'dissolve',
                'duration': 0.4,
                'reason': 'varying_pacing',
                'suitability': 0.6
            })

        # 基于事件类型的建议
        if any(e.type == "scene_start" for e in events):
            hooks['transition_suggestions'].append({
                'type': 'hard_cut',
                'reason': 'scene_change',
                'suitability': 0.9
            })

        return hooks

    def _calculate_quality_score(self, events: List['TimelineEvent'],
                                 duration: float) -> float:
        """
        计算分段质量分数

        参数:
            events: 事件列表
            duration: 分段时长

        返回:
            质量分数 (0-1)
        """
        if not events:
            return 0.5  # 无事件的分段质量较低

        # 基础分数
        score = 0.7

        # 1. 事件丰富度加分
        event_types = set(e.type for e in events)
        if len(event_types) >= 2:
            score += 0.1

        # 2. 对话存在加分
        if any(e.type == "dialogue" for e in events):
            score += 0.1

        # 3. 时长合理性
        target_duration = self.config.target_segment_duration
        duration_diff = abs(duration - target_duration)
        if duration_diff <= 1.0:
            score += 0.1
        elif duration_diff > 3.0:
            score -= 0.1

        # 4. 关键元素加分
        key_elements = self._extract_key_elements(events, 0, duration)
        if len(key_elements) >= 2:
            score += 0.1

        # 确保分数在合理范围内
        return max(0.3, min(1.0, score))

    def analyze_segmentation_quality(self, segments: List[TimeSegment]) -> Dict[str, Any]:
        """
        分析分段质量

        参数:
            segments: 分段列表

        返回:
            质量分析报告
        """
        if not segments:
            return {"error": "没有分段数据"}

        total_segments = len(segments)
        total_duration = sum(s.duration for s in segments)

        # 统计时长分布
        durations = [s.duration for s in segments]
        avg_duration = sum(durations) / total_segments
        min_duration = min(durations)
        max_duration = max(durations)

        # 节奏分布
        pacing_dist = {}
        for segment in segments:
            pacing = segment.pacing
            pacing_dist[pacing] = pacing_dist.get(pacing, 0) + 1

        # 质量分数分布
        quality_scores = [s.quality_score for s in segments]
        avg_quality = sum(quality_scores) / total_segments
        quality_dist = {
            'excellent': len([s for s in segments if s.quality_score >= 0.9]),
            'good': len([s for s in segments if 0.7 <= s.quality_score < 0.9]),
            'fair': len([s for s in segments if 0.5 <= s.quality_score < 0.7]),
            'poor': len([s for s in segments if s.quality_score < 0.5])
        }

        return {
            'summary': {
                'total_segments': total_segments,
                'total_duration': total_duration,
                'average_duration': avg_duration,
                'min_duration': min_duration,
                'max_duration': max_duration,
                'average_quality': avg_quality
            },
            'pacing_distribution': pacing_dist,
            'quality_distribution': quality_dist,
            'segments_by_quality': {
                quality: [s.segment_id for s in segments
                          if (quality == 'excellent' and s.quality_score >= 0.9) or
                          (quality == 'good' and 0.7 <= s.quality_score < 0.9) or
                          (quality == 'fair' and 0.5 <= s.quality_score < 0.7) or
                          (quality == 'poor' and s.quality_score < 0.5)]
                for quality in ['excellent', 'good', 'fair', 'poor']
            },
            'recommendations': self._generate_quality_recommendations(segments, quality_dist)
        }

    def _generate_quality_recommendations(self, segments: List[TimeSegment],
                                          quality_dist: Dict) -> List[str]:
        """生成质量改进建议"""
        recommendations = []

        # 检查过长或过短的分段
        long_segments = [s for s in segments if s.duration > self.config.max_segment_duration]
        short_segments = [s for s in segments if s.duration < self.config.min_segment_duration]

        if long_segments:
            recommendations.append(
                f"发现 {len(long_segments)} 个过长分段（> {self.config.max_segment_duration}秒），建议进一步分割"
            )

        if short_segments:
            recommendations.append(
                f"发现 {len(short_segments)} 个过短分段（< {self.config.min_segment_duration}秒），建议合并或扩展"
            )

        # 检查质量较差的分段
        poor_quality = quality_dist.get('poor', 0)
        if poor_quality > 0:
            recommendations.append(
                f"发现 {poor_quality} 个低质量分段（质量分<0.5），建议重新分析这些分段的内容"
            )

        # 检查节奏变化
        pacing_counts = {}
        for segment in segments:
            pacing_counts[segment.pacing] = pacing_counts.get(segment.pacing, 0) + 1

        if pacing_counts.get('varying', 0) > len(segments) * 0.3:
            recommendations.append("节奏变化过多的分段较多，建议优化节奏一致性")

        if len(recommendations) == 0:
            recommendations.append("分段质量良好，无需特别优化")

        return recommendations

    def _optimize_segment_boundaries(self, segments: List[TimeSegment],
                                     timeline_events: List['TimelineEvent']) -> List[TimeSegment]:
        """
        优化分段边界（简化实现）

        参数:
            segments: 初始分段列表
            timeline_events: 所有时间线事件

        返回:
            优化后的分段列表
        """
        # 这里实现边界优化逻辑
        # 由于时间限制，这里返回原始分段
        # 实际应用中应该实现完整的边界优化逻辑
        return segments

    def _split_long_segments(self, segments: List[TimeSegment],
                             timeline_events: List['TimelineEvent']) -> List[TimeSegment]:
        """
        分割长段落（简化实现）

        参数:
            segments: 分段列表
            timeline_events: 时间线事件列表

        返回:
            分割后的分段列表
        """
        # 这里实现长段落分割逻辑
        # 由于时间限制，这里返回原始分段
        # 实际应用中应该实现完整的长段落分割逻辑
        return segments

    def _validate_segment_durations(self, segments: List[TimeSegment]) -> List[TimeSegment]:
        """
        验证分段时长（简化实现）

        参数:
            segments: 分段列表

        返回:
            验证后的分段列表
        """
        return segments

    def _add_continuity_anchors(self, segments: List[TimeSegment]) -> List[TimeSegment]:
        """
        添加连续性锚点（简化实现）

        参数:
            segments: 分段列表

        返回:
            添加了连续性锚点的分段列表
        """
        return segments
