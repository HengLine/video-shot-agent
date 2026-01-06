"""
@FileName: shot_qa_qa.py
@Description: 
@Author: HengLine
@Time: 2026/1/6 16:52
"""
from hengline.agent.continuity_guardian.model.continuity_guardian_report import AnchoredTimeline
from hengline.agent.shot_generator.model.shot_models import SoraReadyShots
from hengline.agent.shot_qa.auto_fix_suggester import AutoFixSuggester
from hengline.agent.shot_qa.model.fix_models import AutoFixSuggestion, ManualFixRecommendation, FixType
from hengline.agent.shot_qa.model.issue_models import IssueSeverity, CriticalIssue, IssueWarning
from hengline.agent.shot_qa.model.review_models import QualityReviewInput, ReviewConfig, QualityThresholds
from hengline.agent.shot_qa_agent import QAReviewAgent


# 测试智能体5
def test_agent5():
    """测试质量审查智能体"""

    print("=" * 60)
    print("智能体5测试：质量审查智能体")
    print("=" * 60)

    # 1. 创建智能体
    agent = QAReviewAgent()

    # 2. 创建测试输入
    test_input = create_test_input()

    # 3. 执行审查
    result = agent.review(test_input)

    # 4. 验证结果
    assert result is not None, "审查结果为空"
    assert hasattr(result, 'final_decision'), "缺少最终决策"
    assert hasattr(result, 'quality_scores'), "缺少质量评分"

    print("✓ 基本功能测试通过")

    # 5. 输出关键信息
    print(f"\n审查结果:")
    print(f"  最终决策: {result.final_decision.decision.value}")
    print(f"  总体质量: {result.quality_scores.overall_quality_score:.2%}")
    print(f"  关键问题: {len(result.critical_issues)} 个")
    print(f"  自动修复建议: {len(result.auto_fix_suggestions)} 个")

    # 6. 检查决策逻辑
    approval_status = result.get_approval_status()
    print(f"\n批准状态:")
    print(f"  是否批准: {approval_status['is_approved']}")
    print(f"  需要修订: {approval_status['requires_revision']}")
    print(f"  需要重新生成: {approval_status['requires_regeneration']}")

    if approval_status['blocking_issues']:
        print(f"  阻塞问题: {approval_status['blocking_issues'][:3]}")

    # 7. 检查修复建议
    if result.auto_fix_suggestions:
        print(f"\n自动修复建议示例:")
        for i, fix in enumerate(result.auto_fix_suggestions[:2], 1):
            print(f"  {i}. {fix.description}")
            print(f"     类型: {fix.fix_type.value}")
            print(f"     置信度: {fix.confidence_score:.2%}")

    # 8. 检查下一步行动
    if result.next_steps:
        print(f"\n下一步行动:")
        for step in result.next_steps[:2]:
            print(f"  • {step.description}")
            if step.action_items:
                print(f"    行动项: {step.action_items[0]}")

    print("\n" + "=" * 60)
    print("智能体5测试完成！")
    print("=" * 60)


def create_test_input():
    """创建测试输入数据"""
    # 这里创建简化的测试数据
    # 实际使用时需要智能体3和智能体4的真实输出

    return QualityReviewInput(
        sora_shots=create_mock_sora_shots(),
        anchored_timeline=create_mock_anchored_timeline(),
        review_config=ReviewConfig(
            strictness_level="balanced",
            enable_continuity_checks=True,
            enable_constraint_checks=True,
            enable_visual_quality_checks=True,
            enable_technical_checks=True
        ),
        quality_thresholds=QualityThresholds()
    )


def create_mock_sora_shots():
    """创建模拟Sora镜头数据"""

    # 简化实现
    class MockShot:
        def __init__(self, shot_id):
            self.shot_id = shot_id
            self.segment_id = f"s{shot_id.split('_')[0][1:]}"
            self.time_range = (0.0, 5.0)
            self.duration = 5.0
            self.full_sora_prompt = f"A cinematic shot of character in a room with natural lighting. {shot_id}"
            self.camera_parameters = MockCameraParams()
            self.satisfied_constraints = []

    class MockCameraParams:
        def __init__(self):
            self.shot_size = MockEnum("medium_close_up")
            self.camera_movement = MockEnum("static")
            self.lens_focal_length = 50

    class MockEnum:
        def __init__(self, value):
            self.value = value

    class MockShots:
        def __init__(self):
            self.shot_sequence = [
                MockShot("s001_shot1"),
                MockShot("s002_shot1"),
                MockShot("s003_shot1")
            ]

    mock_shot = MockShots()
    return SoraReadyShots(
        shot_sequence=mock_shot.shot_sequence
    )


def create_mock_anchored_timeline():
    """创建模拟锚点时间线数据"""

    class MockConstraint:
        def __init__(self, constraint_id):
            self.constraint_id = constraint_id
            self.type = "character_appearance"
            self.priority = 8
            self.description = f"Character wearing blue shirt - {constraint_id}"
            self.sora_instruction = f"Character in blue shirt - {constraint_id}"
            self.applicable_segments = ["s001", "s002", "s003"]
            self.is_enforced = True

    class MockSegment:
        def __init__(self, segment_id):
            self.segment_id = segment_id
            self.hard_constraints = [
                MockConstraint(f"constraint_{segment_id}_1"),
                MockConstraint(f"constraint_{segment_id}_2")
            ]

    class MockTimeline:
        def __init__(self):
            self.anchored_segments = [
                MockSegment("s001"),
                MockSegment("s002"),
                MockSegment("s003")
            ]

    return AnchoredTimeline(
        anchored_segments=MockTimeline().anchored_segments
    )

def test_fixed_fix_suggester():
    """测试修正后的修复建议器"""

    print("=" * 60)
    print("测试修正后的修复建议系统")
    print("=" * 60)

    # 创建修复建议器
    suggester = AutoFixSuggester()

    # 创建测试问题
    critical_issue = CriticalIssue(
        issue_id="critical_001",
        issue_type="continuity_position",
        severity=IssueSeverity.CRITICAL,
        title="角色位置跳跃",
        description="角色从前一镜头的左侧突然跳到当前镜头的右侧",
        location="shot_s001_shot1到shot_s002_shot1",
        root_cause="位置连续性未检查",
        impact="观众会注意到不自然的跳跃",
        must_fix=True,
        blocks_approval=True,
        suggested_fix="添加移动动作或调整起始位置",
        fix_effort="medium"
    )

    warning = IssueWarning(
        issue_id="warning_001",
        issue_type="visual_composition",
        severity=IssueSeverity.MEDIUM,
        title="构图不平衡",
        description="主体过于偏左，画面不平衡",
        location="shot_s001_shot1",
        suggestion_type="improvement",
        recommended_action="使用三分法重新构图",
        expected_benefit="提高视觉吸引力",
        suggested_priority="medium"
    )

    # 测试手动修复建议生成
    print("\n1. 测试手动修复建议生成:")
    manual_fixes = suggester.generate_manual_fixes([critical_issue], [warning])

    print(f"  生成手动修复建议: {len(manual_fixes)} 个")

    for i, fix in enumerate(manual_fixes[:2], 1):
        print(f"  建议 {i}: {fix.description}")
        print(f"     类型: {fix.fix_type.value}")
        print(f"     目标问题: {fix.target_issue_id}")
        print(f"     复杂度: {fix.complexity}")
        print(f"     预估时间: {fix.estimated_time_minutes}分钟")

        # 验证返回的是ManualFixRecommendation
        assert isinstance(fix, ManualFixRecommendation), f"建议{i}不是ManualFixRecommendation"
        assert hasattr(fix, 'fix_steps'), f"建议{i}缺少fix_steps属性"
        assert hasattr(fix, 'required_tools'), f"建议{i}缺少required_tools属性"

        print(f"     修复步骤: {len(fix.fix_steps)}步")
        print(f"     所需工具: {', '.join(fix.required_tools[:2])}")

    # 测试修复优先级排序
    print("\n2. 测试修复优先级排序:")

    auto_fix = AutoFixSuggestion(
        fix_id="auto_fix_001",
        fix_type=FixType.AUTO,
        target_issue_id="test_issue_001",
        description="自动调整相机参数",
        can_be_auto_applied=True,
        auto_apply_confidence=0.85,
        expected_effectiveness=0.8
    )

    prioritized = suggester.prioritize_fixes([auto_fix], manual_fixes)

    print(f"  立即自动应用: {len(prioritized['immediate_auto'])}")
    print(f"  高优先级手动: {len(prioritized['high_priority_manual'])}")
    print(f"  中优先级手动: {len(prioritized['medium_priority_manual'])}")
    print(f"  低优先级手动: {len(prioritized['low_priority_manual'])}")
    print(f"  配置调整: {len(prioritized['config_adjustments'])}")

    # 验证分类正确性
    assert len(prioritized['immediate_auto']) == 1, "自动修复分类错误"
    assert len(prioritized['high_priority_manual']) >= 1, "手动修复分类错误"

    print("\n✓ 所有测试通过！")
    print("=" * 60)


# 使用示例
def main():
    """主函数示例"""
    # 创建智能体实例
    agent = QAReviewAgent()

    # 创建示例输入数据
    example_input = create_example_input()

    # 执行审查
    result = agent.review(example_input)

    # 输出结果
    print("\n" + "=" * 60)
    print("质量审查结果摘要:")
    print("=" * 60)

    print(f"最终决策: {result.final_decision.decision.value}")
    print(f"总体质量得分: {result.quality_scores.overall_quality_score:.2%}")
    print(f"关键问题: {len(result.critical_issues)} 个")
    print(f"警告: {len(result.warnings)} 个")
    print(f"建议: {len(result.suggestions)} 个")
    print(f"自动修复建议: {len(result.auto_fix_suggestions)} 个")

    print("\n质量评分:")
    scores_dict = result.quality_scores.to_dict()
    for category, sub_scores in scores_dict.items():
        if category != "overall":
            print(f"  {category}:")
            for key, value in sub_scores.items():
                print(f"    {key}: {value:.2%}")

    print(f"\n总体评分: {scores_dict['overall']['score']:.2%}")

    print("\n下一步行动:")
    for i, step in enumerate(result.next_steps[:3], 1):
        print(f"  {i}. {step.description}")
        if step.action_items:
            print(f"     行动项: {', '.join(step.action_items[:2])}...")

    # 输出批准状态
    approval_status = result.get_approval_status()
    print(f"\n批准状态: {'通过' if approval_status['is_approved'] else '不通过'}")
    if approval_status['requires_revision']:
        print("  需要修订")
    if approval_status['requires_regeneration']:
        print("  需要重新生成")

    if approval_status['blocking_issues']:
        print(f"\n阻塞问题: {', '.join(approval_status['blocking_issues'][:3])}")


def create_example_input() -> QualityReviewInput:
    """创建示例输入数据"""

    # 这里需要创建完整的输入数据
    # 由于代码较长，这里只展示结构
    # 实际使用时需要从智能体3和智能体4获取真实数据

    # 创建简化示例
    return QualityReviewInput(
        sora_shots=None,  # 实际应该是智能体4的输出
        anchored_timeline=None,  # 实际应该是智能体3的输出
        review_config=ReviewConfig(),
        quality_thresholds=QualityThresholds()
    )


if __name__ == "__main__":
    # main()
    test_agent5()
