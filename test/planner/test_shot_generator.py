"""
@FileName: test_shot_generator.py
@Description: 
@Author: HengLine
@Time: 2026/1/17 23:22
"""
import json

from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.shot.shot_generator_manager import ShotGenerationManager
from utils.obj_utils import dict_to_obj


def load_from_json(json_path: str):
    """从JSON文件加载数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return dict_to_obj(data, UnifiedScript)


def test_example_usage():
    """使用示例"""

    # 1. 创建结构化脚本（模拟您已解析的数据）
    script = load_from_json("script_parser_result.json")
    # 2. 创建生成管理器
    manager = ShotGenerationManager()

    # 3. 生成分镜头（混合模式）
    result = manager.generate_shots_for_scene(script)

    # 4. 输出结果
    print(f"场景: {result.scene_id}")
    print(f"生成方法: {result.generation_method}")
    print(f"置信度: {result.confidence}")
    print(f"镜头数量: {result.shot_count}")
    print(f"总时长: {result.total_duration}秒")
    print("\n分镜头列表:")

    for shot in result.shots:
        print(f"  {shot.sequence_number}. [{shot.shot_type.value}] {shot.description}")
        print(f"     时长: {shot.duration_estimate}s, 角色: {', '.join(shot.characters)}")
        if shot.notes:
            print(f"     备注: {shot.notes}")

    # 5. 转换为JSON保存
    result_dict = result.to_dict()
    print(f"\nJSON格式:\n{json.dumps(result_dict, indent=2, ensure_ascii=False)}")
