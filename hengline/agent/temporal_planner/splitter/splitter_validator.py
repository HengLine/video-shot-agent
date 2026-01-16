"""
@FileName: splitter_validator.py
@Description: 分片辅助工具和验证器
@Author: HengLine
@Time: 2026/1/15 0:04
"""
from typing import List, Dict, Any

from hengline.agent.temporal_planner.temporal_planner_model import TimeSegment


class SegmentValidator:
    """片段验证器"""

    @staticmethod
    def validate_segment(segment: TimeSegment, segment_id: str = None) -> List[str]:
        """验证片段的合法性"""
        errors = []

        # 检查基础属性
        if not segment.segment_id:
            errors.append("片段ID为空")

        if segment.duration <= 0:
            errors.append(f"片段时长无效: {segment.duration}")

        if segment.time_range[0] < 0 or segment.time_range[1] < 0:
            errors.append(f"时间范围包含负数: {segment.time_range}")

        if segment.time_range[1] <= segment.time_range[0]:
            errors.append(f"结束时间不大于开始时间: {segment.time_range}")

        # 检查元素
        if not segment.contained_elements:
            errors.append("片段没有包含任何元素")
        else:
            # 检查元素时间重叠
            elements_by_time = []
            for elem in segment.contained_elements:
                if elem.start_offset < 0:
                    errors.append(f"元素 {elem.element_id} 开始偏移为负: {elem.start_offset}")

                if elem.duration <= 0:
                    errors.append(f"元素 {elem.element_id} 时长为负或零: {elem.duration}")

                end_offset = elem.start_offset + elem.duration
                if end_offset > segment.duration + 0.1:  # 允许小误差
                    errors.append(f"元素 {elem.element_id} 超出片段范围: {end_offset} > {segment.duration}")

                elements_by_time.append((elem.start_offset, end_offset, elem.element_id))

            # 检查重叠（允许小重叠，因为可能同时发生）
            elements_by_time.sort()
            for i in range(len(elements_by_time) - 1):
                _, end1, id1 = elements_by_time[i]
                start2, _, id2 = elements_by_time[i + 1]

                if end1 > start2 + 0.5:  # 超过0.5秒的重叠需要警告
                    errors.append(f"元素 {id1} 和 {id2} 有明显重叠")

        return errors

    @staticmethod
    def validate_split_result(result: List[TimeSegment]) -> Dict[str, Any]:
        """验证分片结果"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {
                "segments_checked": 0,
                "elements_checked": 0,
                "errors_found": 0,
                "warnings_found": 0
            }
        }

        # 验证每个片段
        for segment in result:
            errors = SegmentValidator.validate_segment(segment)
            if errors:
                validation_result["valid"] = False
                validation_result["errors"].extend(
                    [f"{segment.segment_id}: {error}" for error in errors]
                )
                validation_result["statistics"]["errors_found"] += len(errors)

            validation_result["statistics"]["segments_checked"] += 1
            validation_result["statistics"]["elements_checked"] += len(segment.contained_elements)

        # 检查时间连续性
        time_errors = SegmentValidator.check_time_continuity(result)
        if time_errors:
            validation_result["warnings"].extend(time_errors)
            validation_result["statistics"]["warnings_found"] += len(time_errors)

        # 检查元素完整性
        completeness_warnings = SegmentValidator.check_element_completeness(result)
        if completeness_warnings:
            validation_result["warnings"].extend(completeness_warnings)
            validation_result["statistics"]["warnings_found"] += len(completeness_warnings)

        return validation_result

    @staticmethod
    def check_time_continuity(segments: List[TimeSegment]) -> List[str]:
        """检查时间连续性"""
        warnings = []

        if not segments:
            return warnings

        segments.sort(key=lambda s: s.start_time)

        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]

            # 检查时间间隙
            gap = next_seg.start_time - current.end_time
            if gap > 0.1:  # 超过0.1秒的间隙
                warnings.append(f"片段 {current.segment_id} 和 {next_seg.segment_id} 之间有 {gap:.2f} 秒间隙")

            # 检查时间重叠（不应该有）
            if current.end_time > next_seg.start_time + 0.1:
                overlap = current.end_time - next_seg.start_time
                warnings.append(f"片段 {current.segment_id} 和 {next_seg.segment_id} 重叠 {overlap:.2f} 秒")

        return warnings

    @staticmethod
    def check_element_completeness(segments: List[TimeSegment]) -> List[str]:
        """检查元素完整性"""
        warnings = []

        # 跟踪部分元素
        partial_elements = {}

        for segment in segments:
            for elem in segment.contained_elements:
                if elem.is_partial:
                    if elem.element_id not in partial_elements:
                        partial_elements[elem.element_id] = []
                    partial_elements[elem.element_id].append((segment.segment_id, elem.partial_type))

        # 检查部分元素是否被正确连接
        for element_id, appearances in partial_elements.items():
            if len(appearances) == 1:
                seg_id, partial_type = appearances[0]
                warnings.append(f"元素 {element_id} 只在片段 {seg_id} 中出现为部分({partial_type})，可能未完成")

            # 检查部分类型序列
            partial_types = [p_type for _, p_type in appearances]
            if "start" in partial_types and "end" not in partial_types:
                warnings.append(f"元素 {element_id} 有开始部分但没有结束部分")
            if "end" in partial_types and "start" not in partial_types:
                warnings.append(f"元素 {element_id} 有结束部分但没有开始部分")

        return warnings


class SegmentVisualizer:
    """片段可视化工具"""

    @staticmethod
    def generate_timeline_visualization(segments: List[TimeSegment],
                                        width: int = 100) -> str:
        """生成时间线可视化文本"""
        if not segments:
            return "没有片段可显示"

        # 找到总时长
        total_duration = max(seg.end_time for seg in segments)

        # 计算缩放因子
        scale = width / total_duration

        visualization = []
        visualization.append("=" * (width + 20))
        visualization.append("时间线可视化 (每个字符代表 {:.1f} 秒)".format(total_duration / width))
        visualization.append("=" * (width + 20))

        for segment in sorted(segments, key=lambda s: s.start_time):
            # 计算位置
            start_pos = int(segment.start_time * scale)
            end_pos = int(segment.end_time * scale)
            seg_width = max(1, end_pos - start_pos)

            # 创建片段条
            if seg_width >= 3:
                # 有足够空间显示ID
                bar = f"[{segment.segment_id[-3:]:^{seg_width - 2}}]"
            else:
                # 空间不足，用简单表示
                bar = "[" + "=" * (seg_width - 2) + "]" if seg_width >= 2 else "#"

            # 添加缩进
            line = " " * start_pos + bar

            # 添加信息
            info = f" {segment.segment_id} ({segment.duration:.1f}s)"
            line = line.ljust(width) + info

            visualization.append(line)

            # 添加元素信息（简略）
            if len(segment.contained_elements) <= 3:
                elem_str = " | ".join(
                    f"{elem.element_type.value[0]}:{elem.element_id[-3:]}"
                    for elem in segment.contained_elements[:3]
                )
                visualization.append(" " * start_pos + "  " + elem_str)

        visualization.append("=" * (width + 20))

        return "\n".join(visualization)

    @staticmethod
    def generate_segment_summary_table(segments: List[TimeSegment]) -> str:
        """生成片段摘要表格"""
        headers = ["ID", "开始", "结束", "时长", "元素数", "类型", "节奏分", "完整度"]
        rows = []

        for seg in sorted(segments, key=lambda s: s.start_time):
            element_count = len(seg.contained_elements)
            partial_count = sum(1 for e in seg.contained_elements if e.is_partial)

            row = [
                seg.segment_id,
                f"{seg.start_time:.1f}s",
                f"{seg.end_time:.1f}s",
                f"{seg.duration:.1f}s",
                f"{element_count}({partial_count}部分)",
                seg.segment_type,
                f"{seg.pacing_score:.1f}",
                f"{seg.completeness_score:.2f}"
            ]
            rows.append(row)

        # 计算列宽
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # 创建表格
        table_lines = []

        # 表头
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        separator = "-+-".join("-" * w for w in col_widths)

        table_lines.append(header_line)
        table_lines.append(separator)

        # 数据行
        for row in rows:
            row_line = " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
            table_lines.append(row_line)

        # 统计信息
        if rows:
            total_duration = sum(seg.duration for seg in segments)
            total_elements = sum(len(seg.contained_elements) for seg in segments)
            avg_pacing = sum(seg.pacing_score for seg in segments) / len(segments)

            stats = [
                f"总计: {len(segments)}个片段",
                f"总时长: {total_duration:.1f}秒",
                f"总元素: {total_elements}个",
                f"平均节奏分: {avg_pacing:.1f}"
            ]
            table_lines.append("-" * len(separator))
            table_lines.append(" | ".join(stats))

        return "\n".join(table_lines)
