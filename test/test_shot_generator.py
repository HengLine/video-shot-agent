"""
@FileName: test_shot_generator.py
@Description: 
@Author: HengLine
@Time: 2026/1/5 23:42
"""
import json

from hengline.agent import ShotGeneratorAgent
from hengline.agent.continuity_guardian.continuity_guardian_model import SegmentState, HardConstraint, AnchoredSegment
from hengline.agent.continuity_guardian.model.continuity_rule_guardian import ContinuityRuleSet
from hengline.agent.continuity_guardian.model.continuity_state_guardian import CharacterState
from hengline.agent.shot_generator.model.data_models import ContinuityAnchoredInput
from hengline.agent.temporal_planner.temporal_planner_model import TimeSegment


# 使用示例
def main():
    """主函数示例"""
    # 创建智能体实例
    agent = ShotGeneratorAgent()

    # 创建示例输入数据（实际应用中从智能体3接收）
    example_input = create_example_input()

    # 处理输入
    result = agent.process(example_input)

    # 输出结果
    print("\n" + "=" * 50)
    print("生成结果摘要:")
    print("=" * 50)
    print(f"镜头数量: {len(result.shot_sequence)}")
    print(f"总时长: {sum(s.duration for s in result.shot_sequence):.1f}秒")
    print(f"约束满足率: {result.constraint_satisfaction:.2%}")
    print(f"视觉吸引力: {result.visual_appeal_score:.2%}")

    # 输出前几个镜头的提示词
    print("\n前3个镜头的提示词:")
    for i, shot in enumerate(result.shot_sequence[:3]):
        print(f"\n镜头 {i + 1} ({shot.shot_id}):")
        print(f"  完整提示词: {shot.full_sora_prompt[:100]}...")
        print(f"  镜头: {shot.camera_parameters.shot_size.value}")
        print(f"  运动: {shot.camera_parameters.camera_movement.value}")

    # 保存到JSON文件
    output_path = "sora_ready_shots.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json_data = result.to_json()
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"\n完整结果已保存到: {output_path}")


def create_example_input() -> ContinuityAnchoredInput:
    """创建示例输入数据"""
    # 创建时间片段
    time_segment = TimeSegment(
        segment_id="s001",
        time_range=(0.0, 5.0),
        duration=5.0,
        visual_content="林然坐在沙发上，手中拿着半满的咖啡杯，微笑着看着李薇",
        continuity_hooks={"from_previous": "start", "to_next": "林然将咖啡杯递给李薇"},
        content_type="dialogue_intimate",
        emotional_tone="happy",
        action_intensity=1.0
    )

    # 创建角色状态
    character_state = CharacterState(
        character_name="林然",
        timestamp=0.0,
        appearance={
            "clothing": {"color": "蓝色", "type": "衬衫"},
            "hairstyle": "短发",
            "makeup": "淡妆",
            "accessories": ["手表"]
        },
        posture={
            "position": "sitting",
            "orientation": 45.0,
            "facial_expression": "smiling",
            "body_parts": {"hand_right": "holding_cup"}
        },
        spatial_position={
            "location": "沙发左侧",
            "coordinates": None,
            "relative_to": ["沙发"]
        }
    )

    # 创建片段状态
    segment_state = SegmentState(
        timestamp=0.0,
        segment_id="s001",
        character_states={"林然": character_state},
        prop_states={
            "coffee_cup": {
                "name": "咖啡杯",
                "state": "half_full",
                "location": "林然右手中",
                "material": "陶瓷"
            }
        },
        environment_state={
            "type": "living_room",
            "lighting": "natural_afternoon",
            "time_of_day": "afternoon"
        },
        spatial_relations=[],
        visual_signatures=["林然_蓝色衬衫", "白色咖啡杯", "现代沙发"]
    )

    # 创建硬约束
    hard_constraints = [
        HardConstraint(
            constraint_id="char_appearance_001",
            type="character_appearance",
            priority=9,
            description="林然必须穿着蓝色衬衫，坐在沙发左侧",
            sora_instruction="Lin Ran wearing blue shirt, sitting on left side of sofa",
            applicable_segments=["s001"],
            is_enforced=True
        ),
        HardConstraint(
            constraint_id="prop_state_001",
            type="prop_state",
            priority=8,
            description="白色咖啡杯必须在林然右手中，半满状态",
            sora_instruction="White coffee cup in Lin Ran's right hand, half full",
            applicable_segments=["s001"],
            is_enforced=True
        )
    ]

    # 创建锚点片段
    anchored_segment = AnchoredSegment(
        base_segment=time_segment,
        segment_id="s001",
        hard_constraints=hard_constraints,
        visual_match_requirements={},
        start_state=segment_state,
        end_state=segment_state,  # 简化示例
        keyframes=[],
        transition_to_next={"type": "cut", "duration": 0.3}
    )

    # 创建连续性规则集
    continuity_rules = ContinuityRuleSet(
        global_rules=[{"type": "time_linear", "description": "时间线性流动"}],
        character_rules={"林然": [{"type": "appearance_consistent", "description": "外观一致"}]},
        prop_rules={"coffee_cup": [{"type": "state_continuous", "description": "状态连续"}]},
        environment_rules=[{"type": "lighting_consistent", "description": "光照一致"}],
        temporal_rules=[]
    )

    # 创建完整输入
    return ContinuityAnchoredInput(
        anchored_segments=[anchored_segment],
        continuity_rules=continuity_rules,
        state_snapshots={"0.0": segment_state},
        validation_report={"status": "valid", "issues": []},
        metadata={"source": "example", "version": "1.0"}
    )


if __name__ == "__main__":
    main()
