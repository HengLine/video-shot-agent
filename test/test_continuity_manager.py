"""
@FileName: continuity_guardian_manager.py
@Description: 
@Author: HengLine
@Time: 2026/1/5 16:17
"""
from datetime import datetime
from typing import Dict, Any

from hengline.agent.continuity_guardian.analyzer.scene_analyzer import SceneAnalyzer
from hengline.agent.continuity_guardian.analyzer.state_tracking_engine import StateTrackingEngine
from hengline.agent.continuity_guardian.continuity_constraint_generator import ContinuityConstraintGenerator
from hengline.agent.continuity_guardian.continuity_guardian_manager import IntegratedContinuityGuardian
from hengline.agent.continuity_guardian.model.continuity_guard_guardian import GuardianConfig, GuardMode


# 简化版连续性守护器（用于简单场景）
class SimplifiedContinuityGuardian:
    """简化版连续性守护器 - 只包含核心功能"""

    def __init__(self, task_id: str):
        self.task_id = task_id

        # 只初始化必要的组件
        self.state_tracker = StateTrackingEngine()
        self.constraint_generator = ContinuityConstraintGenerator()
        self.scene_analyzer = SceneAnalyzer(
            GuardianConfig(task_id=task_id)
        )

        self.frame_counter = 0

    def process_scene(self, scene_data: Dict) -> Dict[str, Any]:
        """简化处理流程"""
        self.frame_counter += 1

        # 1. 状态跟踪
        for character in scene_data.get("characters", []):
            char_id = character.get("id")
            if char_id:
                state = {"position": character.get("position")}
                if not self.state_tracker.register_entity("character", char_id, state):
                    self.state_tracker.update_entity_state(char_id, state, datetime.now())

        # 2. 生成约束
        constraints = self.constraint_generator.generate_constraints_for_scene(
            scene_data, None, scene_data.get("scene_type", "general")
        )

        # 3. 场景分析
        complexity = self.scene_analyzer.analyze_scene_complexity(scene_data)

        return {
            "frame": self.frame_counter,
            "constraints": len(constraints),
            "complexity": complexity.value,
            "entities_tracked": len(self.state_tracker.entity_registry)
        }


# 使用示例
def demonstrate_integration():
    """演示集成使用"""
    print("集成式连续性守护器演示")
    print("=" * 60)

    # 创建配置
    config = GuardianConfig(
        task_id="123456",
        mode=GuardMode.ADAPTIVE,
        enable_machine_learning=True,
        enable_auto_fix=True
    )

    # 创建集成守护器
    guardian = IntegratedContinuityGuardian("demo_video", config)

    # 示例场景
    demo_scene = {
        "scene_id": "demo_scene_001",
        "scene_type": "action",
        "characters": [
            {
                "id": "hero",
                "name": "英雄",
                "position": [0, 0, 0],
                "velocity": [2, 0, 0],
                "appearance": {"armor": "heavy"},
                "action": "running"
            },
            {
                "id": "enemy",
                "name": "敌人",
                "position": [5, 0, 0],
                "velocity": [-1, 0, 0],
                "action": "attacking"
            }
        ],
        "props": [
            {
                "id": "sword",
                "name": "剑",
                "position": [2, 1, 0],
                "state": "static",
                "physics": {"mass": 2.0, "material": "steel"}
            }
        ],
        "environment": {
            "time_of_day": "day",
            "weather": "clear",
            "lighting": {"intensity": 1.0}
        }
    }

    # 处理场景
    print("处理示例场景...")
    result = guardian.process_scene(demo_scene)

    print(f"\n处理结果:")
    print(f"  帧号: {result['frame_number']}")
    print(f"  约束生成: {result['constraints_generated']} 个")
    print(f"  物理合理性分数: {result['physics_validation']['plausibility_score']:.2f}")
    print(f"  处理时间: {result['processing_time']:.3f}秒")

    # 显示提示
    print(f"\n生成提示 ({len(result['generation_hints'])} 个):")
    for i, hint in enumerate(result['generation_hints'][:3], 1):
        print(f"  {i}. {hint}")

    # 显示推荐
    if result['recommendations']:
        print(f"\n推荐:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")

    # 系统状态
    status = guardian.get_system_status()
    print(f"\n系统状态:")
    print(f"  跟踪实体: {status['state_tracking']['entities_tracked']} 个")
    print(f"  平均处理时间: {status['performance']['avg_processing_time']:.3f}秒")

    print("\n" + "=" * 60)
    print("演示完成")

    return guardian
