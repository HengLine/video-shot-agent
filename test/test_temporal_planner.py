# -*- coding: utf-8 -*-
"""
测试时序规划智能体配置迁移后的功能
"""
import sys
import json
from pathlib import Path
from hengline.agent.temporal_planner_agent import TemporalPlannerAgent
from hengline.config.temporal_planner_config import get_planner_config


# 测试剧本数据
test_script = {
    "title": "测试剧本",
    "scenes": [
        {
            "id": 1,
            "description": "客厅场景",
            "actions": [
                {
                    "character": "小明",
                    "action": "缓缓坐下",
                    "emotion": "疲惫",
                    "dialogue": "今天工作好累啊"
                },
                {
                    "character": "小红",
                    "action": "快速走过来",
                    "emotion": "关心",
                    "dialogue": "要不要喝杯茶？"
                }
            ]
        },
        {
            "id": 2,
            "description": "厨房场景",
            "actions": [
                {
                    "character": "小红",
                    "action": "准备茶水",
                    "appearance": {"type": "default"}
                },
                {
                    "character": "小明",
                    "action": "思考",
                    "emotion": "沉思"
                }
            ]
        }
    ]
}


def test_temporal_planner():
    """测试时序规划智能体"""
    print("开始测试时序规划智能体...")
    
    # 验证配置加载
    config = get_planner_config()
    print(f"配置摘要: {config.get_config_summary()}")
    
    # 初始化智能体
    planner = TemporalPlannerAgent()
    
    # 执行时序规划
    print("执行时序规划...")
    segments = planner.plan_timeline(test_script)
    
    # 输出结果
    print(f"\n规划结果:")
    print(f"生成了 {len(segments)} 个分段")
    
    # 详细输出每个分段
    for i, segment in enumerate(segments, 1):
        print(f"\n分段 {i}:")
        print(f"  时长: {segment['est_duration']:.2f} 秒")
        print(f"  场景ID: {segment['scene_id']}")
        print(f"  动作数量: {len(segment['actions'])}")
        
        # 输出动作详情
        for j, action in enumerate(segment['actions'], 1):
            print(f"    动作{j}: {action.get('character', '未知角色')} - {action.get('action', '')}")
            if 'dialogue' in action and action['dialogue']:
                print(f"      对话: {action['dialogue']}")
            if 'emotion' in action:
                print(f"      情绪: {action['emotion']}")
            if 'appearance' in action:
                print(f"      外观: {action['appearance']}")
    
    # 保存结果到文件
    output_dir = Path("data/output/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "temporal_planner_test.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    print("测试完成！")
    return 0


if __name__ == "__main__":
    sys.exit(test_temporal_planner())