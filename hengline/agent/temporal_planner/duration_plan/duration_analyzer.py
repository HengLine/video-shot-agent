"""
@FileName: duration_analyzer.py
@Description: 节奏分析器
@Author: HengLine
@Time: 2025/12/20 21:50
"""
import random
from typing import List, Dict, Any, Optional

from marshmallow import ValidationError

from hengline.logger import debug, warning, info, error
from hengline.agent.script_parser.script_parser_model import UnifiedScript
from hengline.agent.temporal_planner.temporal_planner_model import TimeSegment, DurationEstimation, PacingAnalysis


class PacingAnalyzer:
    """节奏分析器"""

    def analyze(self, segments: List[TimeSegment], script_input: UnifiedScript) -> PacingAnalysis:
        """分析整体节奏"""
        if not segments:
            return PacingAnalysis(
                overall_pace="medium",
                pace_variation=0.0,
                emotional_arc=[],
                key_moments=[],
                recommendations=[]
            )

        # 计算平均片段时长
        avg_duration = sum(seg.duration for seg in segments) / len(segments)

        # 判断整体节奏
        if avg_duration > 5.5:
            overall_pace = "slow"
        elif avg_duration < 4.5:
            overall_pace = "fast"
        else:
            overall_pace = "medium"

        # 计算节奏变化
        durations = [seg.duration for seg in segments]
        pace_variation = self._calculate_variation(durations)

        # 分析情绪弧线
        emotional_arc = self._analyze_emotional_arc(segments, script_input)

        # 识别关键时刻
        key_moments = self._identify_key_moments(segments, script_input)

        # 生成建议
        recommendations = self._generate_recommendations(segments, overall_pace, pace_variation)

        return PacingAnalysis(
            overall_pace=overall_pace,
            pace_variation=pace_variation,
            emotional_arc=emotional_arc,
            key_moments=key_moments,
            recommendations=recommendations
        )

    def _calculate_variation(self, durations: List[float]) -> float:
        """计算节奏变化程度（0-1）"""
        if len(durations) <= 1:
            return 0.0

        mean = sum(durations) / len(durations)
        variance = sum((x - mean) ** 2 for x in durations) / len(durations)
        std_dev = variance ** 0.5

        # 归一化到0-1范围
        max_expected_std = 2.0  # 预期的最大标准差
        return min(1.0, std_dev / max_expected_std)

    def _analyze_emotional_arc(self, segments: List[TimeSegment], script_input: UnifiedScript) -> List[Dict]:
        """分析情绪弧线"""
        emotional_arc = []

        for i, segment in enumerate(segments):
            # 这里需要从segment中提取情绪信息
            # 简化实现
            emotional_intensity = random.uniform(0.3, 0.8) if i % 3 == 0 else random.uniform(0.5, 1.0)

            emotional_arc.append({
                "segment_id": segment.segment_id,
                "time": segment.time_range[0],
                "emotional_intensity": round(emotional_intensity, 2),
                "emotional_type": self._infer_emotional_type(segment)
            })

        return emotional_arc

    def _infer_emotional_type(self, segment: TimeSegment) -> str:
        """推断情绪类型"""
        # 简化实现，实际需要分析segment内容
        content = segment.visual_content + segment.audio_content

        emotion_keywords = {
            "紧张": ["紧张", "危急", "危险"],
            "悲伤": ["悲伤", "哭泣", "流泪"],
            "欢乐": ["开心", "欢笑", "喜悦"],
            "愤怒": ["愤怒", "生气", "发火"],
            "平静": ["平静", "安静", "宁静"]
        }

        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in content:
                    return emotion

        return "一般"

    def _identify_key_moments(self, segments: List[TimeSegment], script_input: UnifiedScript) -> List[Dict]:
        """识别关键时刻"""
        key_moments = []

        # 识别长时间停顿
        for i, segment in enumerate(segments):
            if segment.duration > 6.0:  # 超过6秒的片段
                key_moments.append({
                    "type": "extended_moment",
                    "segment_id": segment.segment_id,
                    "reason": "长时间镜头，可能是关键情感时刻",
                    "time": segment.time_range[0],
                    "importance": "high" if segment.duration > 7.0 else "medium"
                })

        # 识别快速切换
        if len(segments) >= 3:
            for i in range(1, len(segments) - 1):
                prev_dur = segments[i - 1].duration
                curr_dur = segments[i].duration
                next_dur = segments[i + 1].duration

                if curr_dur < 3.5 and (prev_dur > 5.0 or next_dur > 5.0):
                    key_moments.append({
                        "type": "rapid_cut",
                        "segment_id": segments[i].segment_id,
                        "reason": "快速切换，可能是动作高潮或情绪转折",
                        "time": segments[i].time_range[0],
                        "importance": "medium"
                    })

        return key_moments

    def _generate_recommendations(self, segments: List[TimeSegment], overall_pace: str, pace_variation: float) -> List[str]:
        """生成节奏优化建议"""
        recommendations = []

        # 基于整体节奏的建议
        if overall_pace == "slow":
            recommendations.append("整体节奏较慢，考虑压缩一些长对话或描述")
        elif overall_pace == "fast":
            recommendations.append("整体节奏较快，考虑在某些情感时刻给更多时间")

        # 基于节奏变化的建议
        if pace_variation < 0.2:
            recommendations.append("节奏变化较小，考虑增加一些快慢对比")
        elif pace_variation > 0.7:
            recommendations.append("节奏变化较大，考虑在某些过渡处增加平滑处理")

        # 检查是否有不合理的时长
        for segment in segments:
            if segment.duration < 3.0:
                recommendations.append(f"片段 {segment.segment_id} 过短（{segment.duration:.1f}秒），可能影响叙事连贯性")
            elif segment.duration > 7.0:
                recommendations.append(f"片段 {segment.segment_id} 过长（{segment.duration:.1f}秒），考虑分割")

        return list(set(recommendations))[:5]  # 去重并限制数量


class QualityValidator:
    """质量验证器"""

    def validate(self, segments: List[TimeSegment],
                 estimations: Dict[str, DurationEstimation],
                 script_input: UnifiedScript) -> Dict[str, float]:
        """验证规划质量"""
        metrics = {
            "segment_quality": self._validate_segment_quality(segments),
            "continuity_score": self._validate_continuity(segments),
            "pacing_score": self._validate_pacing(segments),
            "coverage_score": self._validate_coverage(segments, script_input),
            "confidence_score": self._validate_confidence(estimations)
        }

        # 计算总体质量分数
        weights = {
            "segment_quality": 0.3,
            "continuity_score": 0.25,
            "pacing_score": 0.2,
            "coverage_score": 0.15,
            "confidence_score": 0.1
        }

        overall_quality = sum(metrics[key] * weights[key] for key in metrics)
        metrics["overall_quality"] = overall_quality

        return metrics

    def _validate_segment_quality(self, segments: List[TimeSegment]) -> float:
        """验证片段质量"""
        if not segments:
            return 0.0

        scores = []
        for segment in segments:
            # 时长合理性
            duration_score = 1.0 - abs(segment.duration - 5.0) / 2.0

            # 内容充实度
            content_score = min(1.0, (len(segment.visual_content) + len(segment.audio_content)) / 100)

            # 关键元素存在性
            element_score = 1.0 if segment.key_elements else 0.5

            segment_score = (duration_score * 0.4 + content_score * 0.4 + element_score * 0.2)
            scores.append(segment_score)

        return sum(scores) / len(scores)

    def _validate_continuity(self, segments: List[TimeSegment]) -> float:
        """验证连续性"""
        if len(segments) <= 1:
            return 1.0

        continuity_issues = 0

        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]

            # 检查时间连续性
            time_gap = curr.time_range[0] - prev.time_range[1]
            if abs(time_gap) > 0.1:  # 超过0.1秒的间隔
                continuity_issues += 1

        # 计算连续性分数
        max_issues = len(segments) - 1
        if max_issues == 0:
            return 1.0

        return 1.0 - (continuity_issues / max_issues)

    def _validate_pacing(self, segments: List[TimeSegment]) -> float:
        """验证节奏合理性"""
        if len(segments) <= 2:
            return 1.0

        durations = [seg.duration for seg in segments]
        avg_duration = sum(durations) / len(durations)

        # 计算标准差
        variance = sum((d - avg_duration) ** 2 for d in durations) / len(durations)
        std_dev = variance ** 0.5

        # 理想的节奏变化：有一定变化但不剧烈
        ideal_std = 1.0
        std_score = 1.0 - min(1.0, abs(std_dev - ideal_std) / 2.0)

        return std_score

    def _validate_coverage(self, segments: List[TimeSegment], script_input: UnifiedScript) -> None:
        """
        验证时间片段中的覆盖率信息完整性

        参数:
            segments: 时间片段列表
            script_input: 脚本输入对象

        异常:
            ValidationError: 当数据验证失败时抛出
        """
        # 1. 验证输入参数
        if not segments:
            raise ValidationError("时间片段列表为空")

        if not script_input:
            raise ValidationError("脚本输入对象为空")

        debug(f"开始验证覆盖率信息，时间片段数: {len(segments)}")

        # 2. 检查覆盖率数据来源
        # 可能的方式: 从script_input的属性中获取，或从segments中收集
        coverage_data = None

        # 尝试从script_input中获取覆盖率数据
        for attr_name in ['coverage', 'coverage_stats', 'test_coverage']:
            if hasattr(script_input, attr_name):
                coverage_data = getattr(script_input, attr_name)
                debug(f"从script_input.{attr_name}找到覆盖率数据")
                break

        # 如果script_input中没有，尝试从segments中收集
        if coverage_data is None:
            coverage_data = self._extract_coverage_from_segments(segments)
            if coverage_data:
                debug(f"从时间片段中提取到覆盖率数据")

        # 3. 如果没有覆盖率数据，根据需求决定是警告还是错误
        if coverage_data is None:
            warning("未找到覆盖率数据，跳过覆盖率验证")
            return  # 如果覆盖率是可选的，直接返回

        # 4. 验证覆盖率数据结构
        if isinstance(coverage_data, dict):
            self._validate_dict_coverage(coverage_data, segments)
        elif isinstance(coverage_data, list):
            self._validate_list_coverage(coverage_data, segments)
        else:
            # 尝试将其他类型转换为字典或进行基本验证
            self._validate_generic_coverage(coverage_data, segments)

        # 5. 验证时间片段中的覆盖率引用
        self._validate_segment_coverage_references(segments, coverage_data)

        info("覆盖率验证完成")

    def _extract_coverage_from_segments(self, segments: List[TimeSegment]) -> Optional[Dict]:
        """
        从时间片段中提取覆盖率数据

        返回:
            覆盖率数据字典或None
        """
        coverage_items = []

        for segment in segments:
            # 检查segment是否有覆盖率相关信息
            for attr_name in ['coverage_info', 'test_coverage', 'coverage_metrics']:
                if hasattr(segment, attr_name):
                    coverage_info = getattr(segment, attr_name)
                    if coverage_info:
                        coverage_items.append(coverage_info)

            # 检查segment的metadata中是否有覆盖率信息
            if hasattr(segment, 'metadata') and segment.metadata:
                if 'coverage' in segment.metadata:
                    coverage_items.append(segment.metadata['coverage'])
                elif 'test_coverage' in segment.metadata:
                    coverage_items.append(segment.metadata['test_coverage'])

        if not coverage_items:
            return None

        # 合并覆盖率数据
        merged_coverage = self._merge_coverage_data(coverage_items)
        return merged_coverage

    def _merge_coverage_data(self, coverage_items: List[Any]) -> Dict:
        """
        合并多个覆盖率数据项

        返回:
            合并后的覆盖率数据字典
        """
        merged = {
            'overall_coverage': {
                'percent_covered': 0.0,
                'covered_lines': 0,
                'total_lines': 0
            },
            'file_coverage': []
        }

        file_coverage_map = {}

        for item in coverage_items:
            # 处理不同类型的覆盖率数据
            if isinstance(item, dict):
                # 合并整体覆盖率
                if 'percent_covered' in item:
                    merged['overall_coverage']['percent_covered'] = max(
                        merged['overall_coverage']['percent_covered'],
                        item.get('percent_covered', 0)
                    )

                # 处理文件覆盖率
                if 'files' in item and isinstance(item['files'], list):
                    for file_data in item['files']:
                        if isinstance(file_data, dict) and 'filepath' in file_data:
                            filepath = file_data['filepath']
                            if filepath not in file_coverage_map:
                                file_coverage_map[filepath] = file_data.copy()
                            else:
                                # 合并相同文件的覆盖率数据
                                existing = file_coverage_map[filepath]
                                existing['covered_lines'] = max(
                                    existing.get('covered_lines', 0),
                                    file_data.get('covered_lines', 0)
                                )
                                existing['total_lines'] = max(
                                    existing.get('total_lines', 0),
                                    file_data.get('total_lines', 0)
                                )

        # 计算汇总数据
        total_covered = 0
        total_lines = 0

        for filepath, file_data in file_coverage_map.items():
            covered = file_data.get('covered_lines', 0)
            total = file_data.get('total_lines', 0)

            # 计算文件覆盖率百分比
            if total > 0:
                file_data['percent_covered'] = (covered / total) * 100
            else:
                file_data['percent_covered'] = 0.0

            merged['file_coverage'].append(file_data)

            total_covered += covered
            total_lines += total

        # 计算整体覆盖率
        if total_lines > 0:
            merged['overall_coverage']['percent_covered'] = (total_covered / total_lines) * 100

        merged['overall_coverage']['covered_lines'] = total_covered
        merged['overall_coverage']['total_lines'] = total_lines

        return merged

    def _validate_dict_coverage(self, coverage_data: Dict, segments: List[TimeSegment]) -> None:
        """验证字典类型的覆盖率数据"""
        # 检查基本结构
        if not coverage_data:
            raise ValidationError("覆盖率数据字典为空")

        # 验证整体覆盖率
        overall = coverage_data.get('overall_coverage', {})
        if overall:
            percent = overall.get('percent_covered', 0)
            if percent < 0 or percent > 100:
                raise ValidationError(f"整体覆盖率百分比超出范围: {percent}")

        # 验证文件覆盖率
        file_coverage = coverage_data.get('file_coverage', [])
        if file_coverage and not isinstance(file_coverage, list):
            raise ValidationError(f"文件覆盖率数据应为列表类型，实际为 {type(file_coverage).__name__}")

        # 验证每个文件的数据
        for i, file_data in enumerate(file_coverage if file_coverage else []):
            if not isinstance(file_data, dict):
                warning(f"文件覆盖率数据[{i}]不是字典类型: {type(file_data).__name__}")
                continue

            filepath = file_data.get('filepath', '')
            if not filepath:
                warning(f"文件覆盖率数据[{i}]缺少文件路径")
                continue

            percent = file_data.get('percent_covered', 0)
            if percent < 0 or percent > 100:
                raise ValidationError(f"文件 {filepath} 的覆盖率百分比超出范围: {percent}")

    def _validate_object_coverage(self, coverage_obj: Any, segments: List[TimeSegment]) -> None:
        """
        验证对象类型的覆盖率数据

        参数:
            coverage_obj: 覆盖率数据对象
            segments: 时间片段列表，用于上下文验证
        """
        if coverage_obj is None:
            raise ValidationError("覆盖率对象为空")

        # 记录对象类型以便调试
        obj_type = type(coverage_obj).__name__
        debug(f"验证对象类型覆盖率数据: {obj_type}")

        # 1. 检查对象是否有必要的属性或方法
        required_attrs = []

        # 根据常见覆盖率对象结构检查属性
        coverage_attrs_to_check = [
            'percent_covered',  # 覆盖率百分比
            'coverage_percent',  # 另一种命名
            'covered_lines',  # 覆盖行数
            'total_lines',  # 总行数
            'files',  # 文件列表
            'file_coverage',  # 文件覆盖率
            'overall',  # 整体数据
            'summary'  # 摘要数据
        ]

        available_attrs = []
        for attr in coverage_attrs_to_check:
            if hasattr(coverage_obj, attr):
                available_attrs.append(attr)

        if not available_attrs:
            # 检查对象是否有__dict__属性
            if hasattr(coverage_obj, '__dict__'):
                obj_dict = coverage_obj.__dict__
                if obj_dict:
                    debug(f"使用对象字典进行验证: {list(obj_dict.keys())}")
                    self._validate_dict_coverage(obj_dict, segments)
                    return
                else:
                    raise ValidationError(f"覆盖率对象 {obj_type} 没有可验证的属性")
            else:
                raise ValidationError(f"覆盖率对象 {obj_type} 没有可验证的属性或方法")

        debug(f"覆盖率对象可用属性: {available_attrs}")

        # 2. 验证覆盖率百分比
        percent_attrs = ['percent_covered', 'coverage_percent', 'coverage']
        percent_value = None

        for attr in percent_attrs:
            if hasattr(coverage_obj, attr):
                value = getattr(coverage_obj, attr)
                if isinstance(value, (int, float)):
                    percent_value = value
                    break

        if percent_value is not None:
            if percent_value < 0 or percent_value > 100:
                raise ValidationError(f"覆盖率对象 {obj_type} 的百分比值超出范围: {percent_value}")

            # 记录验证通过
            debug(f"覆盖率对象百分比验证通过: {percent_value:.2f}%")
        else:
            warning(f"覆盖率对象 {obj_type} 未找到百分比属性")

        # 3. 验证行数数据
        if hasattr(coverage_obj, 'covered_lines') and hasattr(coverage_obj, 'total_lines'):
            covered = coverage_obj.covered_lines
            total = coverage_obj.total_lines

            if not isinstance(covered, int) or covered < 0:
                raise ValidationError(f"覆盖率对象 {obj_type} 的覆盖行数无效: {covered}")

            if not isinstance(total, int) or total <= 0:
                raise ValidationError(f"覆盖率对象 {obj_type} 的总行数无效: {total}")

            if covered > total:
                raise ValidationError(f"覆盖率对象 {obj_type}: 覆盖行数({covered})不能大于总行数({total})")

            # 验证百分比与行数的一致性
            if percent_value is not None and total > 0:
                calculated_percent = (covered / total) * 100
                if abs(calculated_percent - percent_value) > 0.01:
                    warning(
                        f"覆盖率对象 {obj_type} 百分比不一致: "
                        f"计算值={calculated_percent:.2f}%, 对象值={percent_value:.2f}%"
                    )

        # 4. 验证文件覆盖率数据
        files_attr = None
        for attr in ['files', 'file_coverage', 'file_data']:
            if hasattr(coverage_obj, attr):
                files_attr = attr
                break

        if files_attr:
            files_data = getattr(coverage_obj, files_attr)

            if files_data is not None:
                if isinstance(files_data, list):
                    self._validate_file_coverage_list(files_data, obj_type)
                elif isinstance(files_data, dict):
                    self._validate_file_coverage_dict(files_data, obj_type)
                elif hasattr(files_data, '__iter__'):
                    # 处理可迭代对象
                    files_list = list(files_data)
                    self._validate_file_coverage_list(files_list, obj_type)
                else:
                    warning(f"覆盖率对象 {obj_type} 的文件数据格式不支持: {type(files_data).__name__}")

        # 5. 检查对象是否有验证方法
        if hasattr(coverage_obj, 'validate'):
            try:
                coverage_obj.validate()
                debug(f"覆盖率对象 {obj_type} 的自定义验证通过")
            except Exception as e:
                raise ValidationError(f"覆盖率对象 {obj_type} 自定义验证失败: {str(e)}")

        # 6. 验证对象与时间片段的关联
        self._validate_object_segment_association(coverage_obj, segments, obj_type)

        debug(f"覆盖率对象 {obj_type} 验证完成")

    def _validate_file_coverage_list(self, files_list: List, source_type: str) -> None:
        """验证文件覆盖率列表"""
        if not files_list:
            warning(f"{source_type} 的文件列表为空")
            return

        total_files = len(files_list)
        valid_files = 0

        for i, file_item in enumerate(files_list):
            try:
                if isinstance(file_item, dict):
                    # 验证字典类型的文件数据
                    self._validate_file_dict(file_item, i, source_type)
                    valid_files += 1
                elif hasattr(file_item, '__dict__'):
                    # 验证对象类型的文件数据
                    self._validate_file_object(file_item, i, source_type)
                    valid_files += 1
                else:
                    warning(f"{source_type} 文件数据[{i}]类型不支持: {type(file_item).__name__}")
            except ValidationError as e:
                raise ValidationError(f"{source_type} 文件数据[{i}]验证失败: {str(e)}")
            except Exception as e:
                error(f"{source_type} 文件数据[{i}]验证异常: {str(e)}")

        debug(f"{source_type} 文件验证: {valid_files}/{total_files} 个文件有效")

    def _validate_file_coverage_dict(self, files_dict: Dict, source_type: str) -> None:
        """验证文件覆盖率字典"""
        if not files_dict:
            warning(f"{source_type} 的文件字典为空")
            return

        total_files = len(files_dict)
        valid_files = 0

        for filepath, file_data in files_dict.items():
            try:
                if not isinstance(filepath, str) or not filepath:
                    raise ValidationError(f"文件路径无效: {filepath}")

                if isinstance(file_data, dict):
                    file_data['filepath'] = filepath
                    self._validate_file_dict(file_data, filepath, source_type)
                    valid_files += 1
                elif hasattr(file_data, '__dict__'):
                    # 确保文件对象有filepath属性
                    if not hasattr(file_data, 'filepath'):
                        file_data.filepath = filepath
                    self._validate_file_object(file_data, filepath, source_type)
                    valid_files += 1
                else:
                    warning(f"{source_type} 文件 {filepath} 数据格式不支持: {type(file_data).__name__}")
            except ValidationError as e:
                raise ValidationError(f"{source_type} 文件 {filepath} 验证失败: {str(e)}")
            except Exception as e:
                error(f"{source_type} 文件 {filepath} 验证异常: {str(e)}")

        debug(f"{source_type} 文件字典验证: {valid_files}/{total_files} 个文件有效")

    def _validate_file_dict(self, file_dict: Dict, identifier: Any, source_type: str) -> None:
        """验证字典类型的文件数据"""
        if 'filepath' not in file_dict:
            file_dict['filepath'] = str(identifier)

        filepath = file_dict['filepath']

        # 验证基本字段
        for key in ['percent_covered', 'coverage_percent']:
            if key in file_dict:
                percent = file_dict[key]
                if not isinstance(percent, (int, float)):
                    raise ValidationError(f"文件 {filepath} 的覆盖率百分比类型错误: {type(percent).__name__}")

                if percent < 0 or percent > 100:
                    raise ValidationError(f"文件 {filepath} 的覆盖率百分比超出范围: {percent}")

        # 验证行数数据
        if 'covered_lines' in file_dict and 'total_lines' in file_dict:
            covered = file_dict['covered_lines']
            total = file_dict['total_lines']

            if not isinstance(covered, int) or covered < 0:
                raise ValidationError(f"文件 {filepath} 的覆盖行数无效: {covered}")

            if not isinstance(total, int) or total <= 0:
                raise ValidationError(f"文件 {filepath} 的总行数无效: {total}")

            if covered > total:
                raise ValidationError(f"文件 {filepath}: 覆盖行数({covered})不能大于总行数({total})")

            # 验证一致性
            if 'percent_covered' in file_dict and total > 0:
                calculated = (covered / total) * 100
                reported = file_dict['percent_covered']
                if abs(calculated - reported) > 0.01:
                    warning(
                        f"文件 {filepath} 百分比不一致: "
                        f"计算值={calculated:.2f}%, 报告值={reported:.2f}%"
                    )

    def _validate_file_object(self, file_obj: Any, identifier: Any, source_type: str) -> None:
        """验证对象类型的文件数据"""
        filepath = getattr(file_obj, 'filepath', str(identifier))

        # 验证百分比
        for attr in ['percent_covered', 'coverage_percent', 'coverage']:
            if hasattr(file_obj, attr):
                percent = getattr(file_obj, attr)
                if isinstance(percent, (int, float)):
                    if percent < 0 or percent > 100:
                        raise ValidationError(f"文件对象 {filepath} 的覆盖率百分比超出范围: {percent}")
                    break

        # 验证行数
        if hasattr(file_obj, 'covered_lines') and hasattr(file_obj, 'total_lines'):
            covered = file_obj.covered_lines
            total = file_obj.total_lines

            if not isinstance(covered, int) or covered < 0:
                raise ValidationError(f"文件对象 {filepath} 的覆盖行数无效: {covered}")

            if not isinstance(total, int) or total <= 0:
                raise ValidationError(f"文件对象 {filepath} 的总行数无效: {total}")

            if covered > total:
                raise ValidationError(f"文件对象 {filepath}: 覆盖行数({covered})不能大于总行数({total})")

    def _validate_object_segment_association(self, coverage_obj: Any, segments: List[TimeSegment], obj_type: str) -> None:
        """验证覆盖率对象与时间片段的关联"""
        # 检查覆盖率对象是否有与时间片段关联的信息
        segment_refs = []

        # 可能包含关联信息的属性
        ref_attrs = ['segment_ids', 'related_segments', 'time_segments', 'context']

        for attr in ref_attrs:
            if hasattr(coverage_obj, attr):
                refs = getattr(coverage_obj, attr)
                if refs:
                    if isinstance(refs, list):
                        segment_refs.extend(refs)
                    elif isinstance(refs, (str, int)):
                        segment_refs.append(refs)
                    break

        # 如果有关联信息，验证这些时间片段是否存在
        if segment_refs:
            valid_refs = []
            segment_ids = [getattr(s, 'id', None) for s in segments]

            for ref in segment_refs:
                if ref in segment_ids:
                    valid_refs.append(ref)
                else:
                    warning(f"覆盖率对象 {obj_type} 引用了不存在的时间片段: {ref}")

            if valid_refs:
                debug(f"覆盖率对象 {obj_type} 关联到 {len(valid_refs)} 个时间片段: {valid_refs}")
            else:
                warning(f"覆盖率对象 {obj_type} 的所有时间片段引用都无效")
        else:
            # 如果没有明确的关联，尝试基于时间戳或其他属性建立关联
            self._infer_object_segment_association(coverage_obj, segments, obj_type)

    def _infer_object_segment_association(self, coverage_obj: Any, segments: List[TimeSegment], obj_type: str) -> None:
        """推断覆盖率对象与时间片段的关联"""
        # 尝试基于时间戳推断关联
        obj_timestamp = None

        # 检查对象是否有时间戳属性
        for attr in ['timestamp', 'created_at', 'time', 'start_time']:
            if hasattr(coverage_obj, attr):
                obj_timestamp = getattr(coverage_obj, attr)
                break

        if obj_timestamp and segments:
            # 找到时间上最接近的时间片段
            closest_segment = None
            min_time_diff = float('inf')

            for segment in segments:
                segment_time = getattr(segment, 'timestamp', None)
                if segment_time and isinstance(segment_time, (int, float)) and isinstance(obj_timestamp, (int, float)):
                    time_diff = abs(segment_time - obj_timestamp)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_segment = segment

            if closest_segment and min_time_diff < 3600:  # 1小时内的阈值
                segment_id = getattr(closest_segment, 'id', 'unknown')
                debug(f"推断覆盖率对象 {obj_type} 关联到时间片段 {segment_id} (时间差: {min_time_diff:.0f}秒)")

    def _validate_list_coverage(self, coverage_data: List, segments: List[TimeSegment]) -> None:
        """验证列表类型的覆盖率数据"""
        if not coverage_data:
            raise ValidationError("覆盖率数据列表为空")

        for i, item in enumerate(coverage_data):
            if isinstance(item, dict):
                # 验证字典项
                self._validate_dict_coverage(item, segments)
            elif hasattr(item, '__dict__'):
                # 验证对象项
                self._validate_object_coverage(item, segments)
            else:
                warning(f"覆盖率数据项[{i}]类型未知: {type(item).__name__}")

    def _validate_generic_coverage(self, coverage_data: Any, segments: List[TimeSegment]) -> None:
        """验证通用类型的覆盖率数据"""
        # 尝试将对象转换为字典进行验证
        if hasattr(coverage_data, '__dict__'):
            data_dict = coverage_data.__dict__
            self._validate_dict_coverage(data_dict, segments)
        elif hasattr(coverage_data, 'to_dict'):
            # 如果对象有to_dict方法
            data_dict = coverage_data.to_dict()
            self._validate_dict_coverage(data_dict, segments)
        else:
            warning(f"无法验证的覆盖率数据类型: {type(coverage_data).__name__}")

    def _validate_segment_coverage_references(self, segments: List[TimeSegment], coverage_data: Any) -> None:
        """验证时间片段中的覆盖率引用"""
        coverage_related_count = 0

        for i, segment in enumerate(segments):
            # 检查segment是否有覆盖率相关属性
            has_coverage_reference = False

            # 检查直接属性
            for attr_name in ['coverage_info', 'test_coverage', 'coverage_metrics', 'coverage']:
                if hasattr(segment, attr_name):
                    value = getattr(segment, attr_name)
                    if value is not None:
                        has_coverage_reference = True
                        break

            # 检查metadata
            if not has_coverage_reference and hasattr(segment, 'metadata'):
                metadata = segment.metadata
                if metadata and isinstance(metadata, dict):
                    for key in ['coverage', 'test_coverage', 'coverage_percent']:
                        if key in metadata and metadata[key] is not None:
                            has_coverage_reference = True
                            break

            if has_coverage_reference:
                coverage_related_count += 1

                # 验证覆盖率数据的有效性
                self._validate_segment_coverage_data(segment)

        # 记录统计信息
        if coverage_related_count == 0:
            warning("没有时间片段包含覆盖率信息")
        else:
            debug(f"{coverage_related_count}/{len(segments)} 个时间片段包含覆盖率信息")

    def _validate_segment_coverage_data(self, segment: TimeSegment) -> None:
        """验证单个时间片段的覆盖率数据"""
        # 这里可以添加针对segment的具体验证逻辑
        # 例如验证覆盖率百分比范围、文件引用等

        # 示例验证：检查覆盖率百分比是否在合理范围内
        coverage_value = None

        # 尝试获取覆盖率值
        for attr_name in ['coverage_percent', 'test_coverage', 'coverage']:
            if hasattr(segment, attr_name):
                value = getattr(segment, attr_name)
                if isinstance(value, (int, float)):
                    coverage_value = value
                    break

        # 检查metadata
        if coverage_value is None and hasattr(segment, 'metadata'):
            metadata = segment.metadata
            if metadata and isinstance(metadata, dict):
                for key in ['coverage', 'test_coverage', 'coverage_percent']:
                    if key in metadata and isinstance(metadata[key], (int, float)):
                        coverage_value = metadata[key]
                        break

        # 验证覆盖率值
        if coverage_value is not None:
            if coverage_value < 0 or coverage_value > 100:
                raise ValidationError(
                    f"时间片段 {getattr(segment, 'id', 'unknown')} 的覆盖率值超出范围: {coverage_value}"
                )

    def _validate_confidence(self, estimations: Dict[str, DurationEstimation]) -> float:
        """验证置信度"""
        if not estimations:
            return 0.0

        confidences = [est.confidence for est in estimations.values() if est.confidence is not None]
        if not confidences:
            return 0.0

        return sum(confidences) / len(confidences)
