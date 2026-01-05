"""
@FileName: test_continuity_guardian.py
@Description: 
@Author: HengLine
@Time: 2026/1/4 22:41
"""
from datetime import datetime, timedelta

from hengline.agent.continuity_guardian.detector.change_detector import ChangeDetector
from hengline.agent.continuity_guardian.detector.consistency_checker import ConsistencyChecker
from hengline.agent.continuity_guardian.analyzer.spatial_Index_analyzer import SpatialIndex
from hengline.agent.continuity_guardian.analyzer.temporal_graph_analyzer import TemporalGraph
from hengline.agent.continuity_guardian.validator.bounding_validator import BoundingVolumeType
from hengline.agent.continuity_guardian.validator.collision_validator import CollisionType, CollisionDetector
from hengline.agent.continuity_guardian.validator.material_validator import MaterialDatabase
from hengline.agent.continuity_guardian.validator.motion_validator import MotionAnalyzer


# 测试函数
def test_collision_detector():
    """测试碰撞检测器"""
    detector = CollisionDetector()

    # 创建测试实体
    entities = [
        {
            "id": "entity1",
            "position": (0, 0, 0),
            "bounding_volume": {
                "type": BoundingVolumeType.SPHERE,
                "radius": 1.0
            }
        },
        {
            "id": "entity2",
            "position": (1.5, 0, 0),
            "bounding_volume": {
                "type": BoundingVolumeType.SPHERE,
                "radius": 1.0
            }
        },
        {
            "id": "entity3",
            "position": (10, 10, 10),
            "bounding_volume": {
                "type": BoundingVolumeType.SPHERE,
                "radius": 1.0
            }
        }
    ]

    collisions = detector.detect_collisions(entities)
    print("碰撞检测结果:")
    for collision in collisions:
        if collision["type"] != CollisionType.NO_COLLISION:
            print(f"  碰撞: {collision['entities'][0]} ↔ {collision['entities'][1]}")
            print(f"  类型: {collision['type'].value}")
            print(f"  穿透深度: {collision['penetration_depth']:.3f}")

    return collisions


def test_motion_analyzer():
    """测试运动分析器"""
    analyzer = MotionAnalyzer()

    # 创建测试轨迹（抛物线）
    trajectory = []
    for t in range(0, 10):
        x = t
        y = -0.5 * t ** 2 + 5 * t  # 抛物线
        z = 0
        trajectory.append((x, y, z))

    analysis = analyzer.analyze_motion(trajectory)
    print("\n运动分析结果:")
    print(f"轨迹类型: {analysis['trajectory_type'].value}")
    print(f"总距离: {analysis['total_distance']:.2f}米")
    print(f"平均速度: {analysis['average_speed']:.2f}米/秒")
    print(f"最大加速度: {analysis['acceleration_profile']['max_acceleration']:.2f}米/秒²")
    print(f"平滑度: {analysis['smoothness_metrics']['overall_smoothness']:.2f}")
    print(f"检测到的模式: {', '.join(analysis['motion_patterns'])}")

    return analysis


def test_material_database():
    """测试材料数据库"""
    db = MaterialDatabase()

    # 获取材料
    steel = db.get_material_properties("steel")
    print(f"\n钢材属性:")
    print(f"  密度: {steel.density} kg/m³")
    print(f"  摩擦系数: {steel.friction}")
    print(f"  恢复系数: {steel.restitution}")

    # 搜索材料
    results = db.search_materials({
        "density_min": 2000,
        "density_max": 3000
    })
    print(f"\n密度在2000-3000 kg/m³之间的材料:")
    for material in results:
        print(f"  {material.name}: {material.density} kg/m³")

    # 推荐材料
    recommendation = db.get_recommended_material({
        "target_density": 800,
        "target_friction": 0.4,
        "density_weight": 0.4,
        "friction_weight": 0.3
    })
    if recommendation:
        print(f"\n推荐材料: {recommendation.name}")
        print(f"  匹配分数计算依据...")

    # 获取材料卡片
    card = db.get_material_card("rubber")
    print(f"\n橡胶材料卡片:\n{card}")

    return db


def test_spatial_index():
    """测试空间索引"""
    index = SpatialIndex(cell_size=2.0)

    # 添加实体
    index.update_position("entity1", 0, 0, 0)
    index.update_position("entity2", 1, 1, 0)
    index.update_position("entity3", 5, 5, 0)
    index.update_position("entity4", 1.2, 0.8, 0)

    # 设置边界体
    index.set_bounding_volume("entity1", BoundingVolumeType.SPHERE, radius=1.0)
    index.set_bounding_volume("entity2", BoundingVolumeType.SPHERE, radius=0.8)

    # 查找附近的实体
    nearby = index.find_nearby_entities("entity1", radius=3.0)
    print(f"\nentity1附近3米内的实体: {nearby}")

    # 获取空间关系
    relationships = index.get_spatial_relationships("entity1")
    print(f"\nentity1的空间关系:")
    for rel in relationships:
        print(f"  与 {rel['entity2']}: 距离 {rel['distance']:.2f}米, 类型 {rel['type']}")

    # 射线投射
    hits = index.ray_cast((0, 5, 0), (0, -1, 0), max_distance=10)
    print(f"\n射线投射结果:")
    for hit in hits:
        print(f"  击中 {hit['entity_id']}, 距离 {hit['distance']:.2f}米")

    # 可视化
    visualization = index.visualize_spatial_distribution()
    print(f"\n空间分布可视化:\n{visualization}")

    return index


def test_change_detector():
    """测试变化检测器"""
    detector = ChangeDetector()

    # 测试状态变化
    old_state = {
        "position": [0, 0, 0],
        "rotation": [0, 0, 0],
        "color": [1, 0, 0],
        "intensity": 0.5,
        "material": "steel"
    }

    new_state = {
        "position": [1, 0, 0],  # 位置变化
        "rotation": [10, 0, 0],  # 旋转变化
        "color": [1, 0.1, 0],  # 轻微颜色变化
        "intensity": 0.8,  # 强度变化
        "material": "steel",  # 材料不变
        "new_property": "added"  # 新增属性
    }

    changes = detector.detect_changes(old_state, new_state)
    print(f"\n变化检测结果:")
    print(f"  改变的属性: {changes['changed_attributes']}")
    print(f"  未改变的属性: {changes['unchanged_attributes']}")
    print(f"  总体变化等级: {changes['overall_change_level']}")
    print(f"  变化分数: {changes['change_score']:.3f}")

    print(f"\n显著变化:")
    for change in changes['significant_changes']:
        print(f"  {change['attribute']}: {change['change']}, 幅度: {change.get('magnitude', 'N/A')}")

    # 设置基线和检测漂移
    detector.set_baseline("test_entity", old_state)
    drift = detector.detect_drift_from_baseline("test_entity", new_state)
    print(f"\n相对于基线的漂移:")
    print(f"  漂移分数: {drift['drift_score']:.3f}")
    print(f"  漂移等级: {drift['drift_level']}")

    return changes


def test_consistency_checker():
    """测试一致性检查器"""
    checker = ConsistencyChecker()

    # 创建测试状态序列
    states = [
        {
            "timestamp": datetime(2024, 1, 1, 10, 0, 0),
            "position": [0, 0, 0],
            "velocity": [0, 0, 0],
            "material": "steel",
            "density": 7800
        },
        {
            "timestamp": datetime(2024, 1, 1, 10, 0, 1),
            "position": [0.5, 0, 0],  # 合理移动
            "velocity": [0.5, 0, 0],
            "material": "steel",
            "density": 7800
        },
        {
            "timestamp": datetime(2024, 1, 1, 10, 0, 3),  # 时间跳过2秒
            "position": [10, 0, 0],  # 瞬移
            "velocity": [5, 0, 0],
            "material": "wood",  # 材料突变
            "density": 600
        }
    ]

    report = checker.check_consistency(states)
    print(f"\n一致性检查报告:")
    print(f"  总体一致性分数: {report['overall_consistency_score']:.3f}")

    print(f"\n规则违反:")
    for rule_name, violations in report['rule_violations'].items():
        if violations:
            print(f"  {rule_name}: {len(violations)} 个违反")
            for violation in violations[:2]:  # 显示前2个
                print(f"    - {violation['issue']}: {violation['details']}")

    print(f"\n不一致性:")
    for inconsistency in report['inconsistencies'][:3]:  # 显示前3个
        print(f"  - {inconsistency['issue']}: {inconsistency['details']}")

    print(f"\n建议:")
    for recommendation in report['recommendations']:
        print(f"  - {recommendation}")

    # 生成完整报告
    full_report = checker.generate_consistency_report()
    print(f"\n完整报告摘要:\n{full_report[:500]}...")

    return report


def test_temporal_graph():
    """测试时间图"""
    graph = TemporalGraph(max_events=100)

    # 添加事件
    base_time = datetime(2024, 1, 1, 10, 0, 0)

    event1 = graph.add_event(
        "entity1",
        base_time,
        {"position": [0, 0, 0], "state": "idle"},
        "state_update"
    )

    event2 = graph.add_event(
        "entity1",
        base_time + timedelta(seconds=1),
        {"position": [1, 0, 0], "state": "moving"},
        "state_update"
    )

    event3 = graph.add_event(
        "entity2",
        base_time + timedelta(seconds=2),
        {"position": [5, 0, 0], "state": "idle"},
        "state_update"
    )

    # 添加关系
    graph.add_relationship(event1, event2, "causes", strength=0.8)

    # 获取事件
    entity_events = graph.get_entity_events("entity1")
    print(f"\nentity1的事件数: {len(entity_events)}")

    # 分析时间模式
    analysis = graph.analyze_temporal_patterns("entity1")
    print(f"\nentity1的时间模式分析:")
    print(f"  总事件数: {analysis['total_events']}")
    print(f"  时间跨度: {analysis['time_span']:.1f}秒")
    print(f"  平均间隔: {analysis['interval_stats']['mean']:.2f}秒")
    print(f"  规律性分数: {analysis['regularity_score']:.2f}")

    # 预测下一个事件
    prediction = graph.predict_next_event("entity1", base_time + timedelta(seconds=5))
    if prediction:
        print(f"\n预测下一个事件:")
        print(f"  预测时间: {prediction['predicted_time']}")
        print(f"  置信度: {prediction['confidence']:.2f}")

    # 获取摘要
    summary = graph.get_temporal_graph_summary()
    print(f"\n时间图摘要:\n{summary}")

    return graph


if __name__ == "__main__":
    print("开始测试辅助类...")
    print("=" * 60)

    # 运行所有测试
    test_results = {}

    try:
        print("\n1. 测试碰撞检测器:")
        test_results["collision"] = test_collision_detector()
    except Exception as e:
        print(f"碰撞检测器测试失败: {e}")

    try:
        print("\n2. 测试运动分析器:")
        test_results["motion"] = test_motion_analyzer()
    except Exception as e:
        print(f"运动分析器测试失败: {e}")

    try:
        print("\n3. 测试材料数据库:")
        test_results["material"] = test_material_database()
    except Exception as e:
        print(f"材料数据库测试失败: {e}")

    try:
        print("\n4. 测试空间索引:")
        test_results["spatial"] = test_spatial_index()
    except Exception as e:
        print(f"空间索引测试失败: {e}")

    try:
        print("\n5. 测试变化检测器:")
        test_results["change"] = test_change_detector()
    except Exception as e:
        print(f"变化检测器测试失败: {e}")

    try:
        print("\n6. 测试一致性检查器:")
        test_results["consistency"] = test_consistency_checker()
    except Exception as e:
        print(f"一致性检查器测试失败: {e}")

    try:
        print("\n7. 测试时间图:")
        test_results["temporal"] = test_temporal_graph()
    except Exception as e:
        print(f"时间图测试失败: {e}")

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print(f"成功测试了 {len(test_results)} 个辅助类")
