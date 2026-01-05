"""
@FileName: collision_validator.py
@Description: 碰撞检测器
@Author: HengLine
@Time: 2026/1/4 21:53
"""
import math
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Tuple

from hengline.agent.continuity_guardian.validator.bounding_validator import BoundingVolumeType, BoundingBox


class CollisionType(Enum):
    """碰撞类型枚举"""
    NO_COLLISION = "no_collision"
    SOFT_COLLISION = "soft_collision"  # 软碰撞（可穿透）
    HARD_COLLISION = "hard_collision"  # 硬碰撞（不可穿透）
    GLANCING_BLOW = "glancing_blow"  # 擦边碰撞
    PENETRATION = "penetration"  # 穿透
    RESTING_CONTACT = "resting_contact"  # 静止接触


class CollisionDetector:
    """碰撞检测器"""

    def __init__(self, collision_margin: float = 0.01):
        self.collision_margin = collision_margin
        self.broad_phase_cache: Dict[str, Any] = {}
        self.narrow_phase_cache: Dict[str, Any] = {}
        self.collision_history: deque = deque(maxlen=1000)

    def detect_collisions(self, entities: List[Dict]) -> List[Dict]:
        """检测所有实体间的碰撞"""
        collisions = []

        # 宽相位检测：快速筛选可能碰撞的实体对
        potential_pairs = self._broad_phase_detection(entities)

        # 窄相位检测：精确检测碰撞
        for entity1, entity2 in potential_pairs:
            collision = self._narrow_phase_detection(entity1, entity2)
            if collision["type"] != CollisionType.NO_COLLISION:
                collisions.append(collision)

                # 记录碰撞历史
                self.collision_history.append({
                    "timestamp": datetime.now(),
                    "entity1": entity1.get("id"),
                    "entity2": entity2.get("id"),
                    "collision_type": collision["type"],
                    "details": collision
                })

        return collisions

    def _broad_phase_detection(self, entities: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """宽相位碰撞检测"""
        potential_pairs = []

        # 使用空间分区（简化版网格法）
        grid_cells = defaultdict(list)
        cell_size = 2.0  # 网格单元格大小

        # 将实体分配到网格单元格
        for entity in entities:
            if "position" in entity and "bounding_volume" in entity:
                pos = entity["position"]
                if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    cell_x = int(pos[0] / cell_size)
                    cell_y = int(pos[1] / cell_size)
                    cell_z = int(pos[2] / cell_size)
                    cell_key = f"{cell_x},{cell_y},{cell_z}"
                    grid_cells[cell_key].append(entity)

        # 检查每个单元格内的实体以及相邻单元格
        for cell_key, cell_entities in grid_cells.items():
            # 同单元格内的实体对
            for i in range(len(cell_entities)):
                for j in range(i + 1, len(cell_entities)):
                    potential_pairs.append((cell_entities[i], cell_entities[j]))

            # 检查相邻单元格（简化：只检查右、前、上三个方向）
            cell_parts = cell_key.split(',')
            if len(cell_parts) == 3:
                cell_x, cell_y, cell_z = map(int, cell_parts)

                # 检查相邻单元格
                for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    neighbor_key = f"{cell_x + dx},{cell_y + dy},{cell_z + dz}"
                    if neighbor_key in grid_cells:
                        for entity1 in cell_entities:
                            for entity2 in grid_cells[neighbor_key]:
                                potential_pairs.append((entity1, entity2))

        return potential_pairs

    def _narrow_phase_detection(self, entity1: Dict, entity2: Dict) -> Dict[str, Any]:
        """窄相位精确碰撞检测"""
        # 检查是否有边界体
        if "bounding_volume" not in entity1 or "bounding_volume" not in entity2:
            return self._create_no_collision_result(entity1, entity2)

        bv1 = entity1["bounding_volume"]
        bv2 = entity2["bounding_volume"]

        # 根据边界体类型进行检测
        collision_result = self._detect_bv_collision(bv1, bv2, entity1, entity2)

        # 如果检测到碰撞，计算碰撞详情
        if collision_result["type"] != CollisionType.NO_COLLISION:
            collision_result = self._calculate_collision_details(
                collision_result, entity1, entity2
            )

        return collision_result

    def _detect_bv_collision(self, bv1: Dict, bv2: Dict,
                             entity1: Dict, entity2: Dict) -> Dict[str, Any]:
        """检测边界体碰撞"""
        bv_type1 = bv1.get("type", BoundingVolumeType.SPHERE)
        bv_type2 = bv2.get("type", BoundingVolumeType.SPHERE)

        # 球体-球体碰撞
        if bv_type1 == BoundingVolumeType.SPHERE and bv_type2 == BoundingVolumeType.SPHERE:
            return self._detect_sphere_sphere_collision(bv1, bv2, entity1, entity2)

        # 盒体-盒体碰撞
        elif bv_type1 == BoundingVolumeType.BOX and bv_type2 == BoundingVolumeType.BOX:
            return self._detect_box_box_collision(bv1, bv2, entity1, entity2)

        # 球体-盒体碰撞
        elif (bv_type1 == BoundingVolumeType.SPHERE and bv_type2 == BoundingVolumeType.BOX or
              bv_type1 == BoundingVolumeType.BOX and bv_type2 == BoundingVolumeType.SPHERE):
            return self._detect_sphere_box_collision(bv1, bv2, entity1, entity2)

        # 默认返回无碰撞
        return self._create_no_collision_result(entity1, entity2)

    def _detect_sphere_sphere_collision(self, sphere1: Dict, sphere2: Dict,
                                        entity1: Dict, entity2: Dict) -> Dict[str, Any]:
        """检测球体-球体碰撞"""
        pos1 = entity1.get("position", [0, 0, 0])
        pos2 = entity2.get("position", [0, 0, 0])
        radius1 = sphere1.get("radius", 0.5)
        radius2 = sphere2.get("radius", 0.5)

        # 计算距离
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        # 检查碰撞
        if distance <= (radius1 + radius2 + self.collision_margin):
            # 计算穿透深度
            penetration = (radius1 + radius2) - distance

            # 计算碰撞法线（从球1指向球2）
            if distance > 0:
                normal = (dx / distance, dy / distance, dz / distance)
            else:
                normal = (1, 0, 0)  # 完全重合时的默认法线

            # 计算碰撞点（在球1表面的点）
            contact_point = (
                pos1[0] + normal[0] * radius1,
                pos1[1] + normal[1] * radius1,
                pos1[2] + normal[2] * radius1
            )

            # 确定碰撞类型
            relative_speed = self._calculate_relative_speed(entity1, entity2)
            if relative_speed < 0.1 and penetration < 0.01:
                collision_type = CollisionType.RESTING_CONTACT
            elif relative_speed > 5.0:
                collision_type = CollisionType.HARD_COLLISION
            elif penetration > radius1 * 0.5 or penetration > radius2 * 0.5:
                collision_type = CollisionType.PENETRATION
            else:
                collision_type = CollisionType.SOFT_COLLISION

            return {
                "type": collision_type,
                "entities": (entity1.get("id"), entity2.get("id")),
                "penetration_depth": penetration,
                "contact_normal": normal,
                "contact_point": contact_point,
                "distance_at_impact": distance,
                "relative_speed": relative_speed
            }

        return self._create_no_collision_result(entity1, entity2)

    def _detect_box_box_collision(self, box1: Dict, box2: Dict,
                                  entity1: Dict, entity2: Dict) -> Dict[str, Any]:
        """检测盒体-盒体碰撞"""
        pos1 = entity1.get("position", [0, 0, 0])
        pos2 = entity2.get("position", [0, 0, 0])

        # 创建边界盒对象
        bbox1 = self._create_bounding_box(box1, pos1)
        bbox2 = self._create_bounding_box(box2, pos2)

        # 检查是否相交
        if bbox1.intersects_box(bbox2):
            # 计算穿透深度和法线（简化版）
            penetration, normal = self._calculate_box_penetration(bbox1, bbox2)

            # 计算接触点（两个盒子的中点）
            center1 = bbox1.get_center()
            center2 = bbox2.get_center()
            contact_point = (
                (center1[0] + center2[0]) / 2,
                (center1[1] + center2[1]) / 2,
                (center1[2] + center2[2]) / 2
            )

            relative_speed = self._calculate_relative_speed(entity1, entity2)

            # 确定碰撞类型
            if relative_speed < 0.1 and penetration < 0.01:
                collision_type = CollisionType.RESTING_CONTACT
            elif relative_speed > 3.0:
                collision_type = CollisionType.HARD_COLLISION
            else:
                collision_type = CollisionType.SOFT_COLLISION

            return {
                "type": collision_type,
                "entities": (entity1.get("id"), entity2.get("id")),
                "penetration_depth": penetration,
                "contact_normal": normal,
                "contact_point": contact_point,
                "distance_at_impact": 0,  # 盒体相交时距离为0
                "relative_speed": relative_speed,
                "overlap_volume": self._calculate_overlap_volume(bbox1, bbox2)
            }

        return self._create_no_collision_result(entity1, entity2)

    def _detect_sphere_box_collision(self, sphere_bv: Dict, box_bv: Dict,
                                     entity_sphere: Dict, entity_box: Dict) -> Dict[str, Any]:
        """检测球体-盒体碰撞"""
        # 确定哪个是球体哪个是盒体
        if sphere_bv.get("type") == BoundingVolumeType.SPHERE:
            sphere_entity = entity_sphere
            box_entity = entity_box
            sphere_data = sphere_bv
            box_data = box_bv
        else:
            sphere_entity = entity_box
            box_entity = entity_sphere
            sphere_data = box_bv
            box_data = sphere_bv

        sphere_pos = sphere_entity.get("position", [0, 0, 0])
        box_pos = box_entity.get("position", [0, 0, 0])
        sphere_radius = sphere_data.get("radius", 0.5)

        # 创建边界盒
        bbox = self._create_bounding_box(box_data, box_pos)

        # 找到盒子上离球心最近的点
        closest_point = (
            max(bbox.min_point[0], min(sphere_pos[0], bbox.max_point[0])),
            max(bbox.min_point[1], min(sphere_pos[1], bbox.max_point[1])),
            max(bbox.min_point[2], min(sphere_pos[2], bbox.max_point[2]))
        )

        # 计算球心到最近点的距离
        dx = sphere_pos[0] - closest_point[0]
        dy = sphere_pos[1] - closest_point[1]
        dz = sphere_pos[2] - closest_point[2]
        distance_sq = dx * dx + dy * dy + dz * dz

        # 检查碰撞
        if distance_sq <= (sphere_radius + self.collision_margin) ** 2:
            distance = math.sqrt(distance_sq)

            if distance > 0:
                # 计算穿透深度和法线
                penetration = sphere_radius - distance
                normal = (dx / distance, dy / distance, dz / distance)
                contact_point = closest_point
            else:
                # 球心在盒子内部
                penetration = sphere_radius
                normal = (0, 1, 0)  # 默认法线向上
                contact_point = sphere_pos

            relative_speed = self._calculate_relative_speed(sphere_entity, box_entity)

            # 确定碰撞类型
            if relative_speed < 0.1 and penetration < 0.01:
                collision_type = CollisionType.RESTING_CONTACT
            elif relative_speed > 4.0:
                collision_type = CollisionType.HARD_COLLISION
            elif penetration > sphere_radius * 0.3:
                collision_type = CollisionType.PENETRATION
            else:
                collision_type = CollisionType.SOFT_COLLISION

            return {
                "type": collision_type,
                "entities": (sphere_entity.get("id"), box_entity.get("id")),
                "penetration_depth": penetration,
                "contact_normal": normal,
                "contact_point": contact_point,
                "distance_at_impact": distance,
                "relative_speed": relative_speed,
                "closest_point_on_box": closest_point
            }

        return self._create_no_collision_result(sphere_entity, box_entity)

    def _create_bounding_box(self, box_data: Dict, position: Tuple[float, float, float]) -> BoundingBox:
        """从数据创建边界盒"""
        if "min_point" in box_data and "max_point" in box_data:
            # 已经提供了最小最大点
            min_point = box_data["min_point"]
            max_point = box_data["max_point"]
        elif "size" in box_data:
            # 提供了尺寸
            size = box_data["size"]
            half_size = (size[0] / 2, size[1] / 2, size[2] / 2)
            min_point = (
                position[0] - half_size[0],
                position[1] - half_size[1],
                position[2] - half_size[2]
            )
            max_point = (
                position[0] + half_size[0],
                position[1] + half_size[1],
                position[2] + half_size[2]
            )
        else:
            # 默认尺寸
            min_point = (position[0] - 0.5, position[1] - 0.5, position[2] - 0.5)
            max_point = (position[0] + 0.5, position[1] + 0.5, position[2] + 0.5)

        return BoundingBox(min_point, max_point)

    def _calculate_box_penetration(self, box1: BoundingBox, box2: BoundingBox) -> Tuple[float, Tuple[float, float, float]]:
        """计算盒体间穿透深度和法线"""
        # 计算在各个轴上的重叠
        overlap_x = min(box1.max_point[0], box2.max_point[0]) - max(box1.min_point[0], box2.min_point[0])
        overlap_y = min(box1.max_point[1], box2.max_point[1]) - max(box1.min_point[1], box2.min_point[1])
        overlap_z = min(box1.max_point[2], box2.max_point[2]) - max(box1.min_point[2], box2.min_point[2])

        # 找到最小重叠的轴
        overlaps = [overlap_x, overlap_y, overlap_z]
        min_overlap = min(overlaps)
        min_axis = overlaps.index(min_overlap)

        # 确定法线方向
        center1 = box1.get_center()
        center2 = box2.get_center()

        if min_axis == 0:  # X轴
            normal = (1 if center1[0] < center2[0] else -1, 0, 0)
        elif min_axis == 1:  # Y轴
            normal = (0, 1 if center1[1] < center2[1] else -1, 0)
        else:  # Z轴
            normal = (0, 0, 1 if center1[2] < center2[2] else -1)

        return min_overlap, normal

    def _calculate_overlap_volume(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """计算重叠体积"""
        # 计算重叠区域
        overlap_min = (
            max(box1.min_point[0], box2.min_point[0]),
            max(box1.min_point[1], box2.min_point[1]),
            max(box1.min_point[2], box2.min_point[2])
        )
        overlap_max = (
            min(box1.max_point[0], box2.max_point[0]),
            min(box1.max_point[1], box2.max_point[1]),
            min(box1.max_point[2], box2.max_point[2])
        )

        # 检查是否有有效重叠
        if (overlap_max[0] > overlap_min[0] and
                overlap_max[1] > overlap_min[1] and
                overlap_max[2] > overlap_min[2]):
            width = overlap_max[0] - overlap_min[0]
            height = overlap_max[1] - overlap_min[1]
            depth = overlap_max[2] - overlap_min[2]
            return width * height * depth

        return 0.0

    def _calculate_relative_speed(self, entity1: Dict, entity2: Dict) -> float:
        """计算相对速度"""
        vel1 = entity1.get("velocity", [0, 0, 0])
        vel2 = entity2.get("velocity", [0, 0, 0])

        if isinstance(vel1, (list, tuple)) and len(vel1) >= 3 and \
                isinstance(vel2, (list, tuple)) and len(vel2) >= 3:
            rel_vel = (
                vel2[0] - vel1[0],
                vel2[1] - vel1[1],
                vel2[2] - vel1[2]
            )
            return math.sqrt(rel_vel[0] ** 2 + rel_vel[1] ** 2 + rel_vel[2] ** 2)

        return 0.0

    def _calculate_collision_details(self, collision_result: Dict,
                                     entity1: Dict, entity2: Dict) -> Dict[str, Any]:
        """计算碰撞详情"""
        # 添加实体信息
        collision_result["entity1_info"] = {
            "id": entity1.get("id"),
            "type": entity1.get("type", "unknown"),
            "material": entity1.get("material", "default"),
            "mass": entity1.get("mass", 1.0)
        }

        collision_result["entity2_info"] = {
            "id": entity2.get("id"),
            "type": entity2.get("type", "unknown"),
            "material": entity2.get("material", "default"),
            "mass": entity2.get("mass", 1.0)
        }

        # 计算冲击力（简化版）
        relative_speed = collision_result.get("relative_speed", 0)
        mass1 = entity1.get("mass", 1.0)
        mass2 = entity2.get("mass", 1.0)

        # 简化冲击力计算
        reduced_mass = (mass1 * mass2) / (mass1 + mass2) if (mass1 + mass2) > 0 else 0
        impulse_magnitude = reduced_mass * relative_speed

        collision_result["impact_force"] = impulse_magnitude
        collision_result["energy_loss"] = self._calculate_energy_loss(
            entity1, entity2, relative_speed
        )

        # 添加时间戳
        collision_result["timestamp"] = datetime.now()

        return collision_result

    def _calculate_energy_loss(self, entity1: Dict, entity2: Dict,
                               relative_speed: float) -> float:
        """计算能量损失"""
        # 基于材料恢复系数计算能量损失
        restitution1 = self._get_material_restitution(entity1.get("material", "default"))
        restitution2 = self._get_material_restitution(entity2.get("material", "default"))

        combined_restitution = (restitution1 + restitution2) / 2

        # 能量损失 = 1 - 恢复系数²
        energy_loss = 1.0 - combined_restitution ** 2

        return max(0.0, min(1.0, energy_loss))

    def _get_material_restitution(self, material_name: str) -> float:
        """获取材料恢复系数"""
        material_properties = {
            "rubber": 0.8,
            "steel": 0.5,
            "wood": 0.4,
            "glass": 0.3,
            "clay": 0.1,
            "default": 0.5
        }
        return material_properties.get(material_name, 0.5)

    def _create_no_collision_result(self, entity1: Dict, entity2: Dict) -> Dict[str, Any]:
        """创建无碰撞结果"""
        return {
            "type": CollisionType.NO_COLLISION,
            "entities": (entity1.get("id"), entity2.get("id")),
            "penetration_depth": 0.0,
            "contact_normal": (0, 0, 0),
            "contact_point": (0, 0, 0),
            "distance_at_impact": self._calculate_distance(entity1, entity2),
            "relative_speed": self._calculate_relative_speed(entity1, entity2),
            "timestamp": datetime.now()
        }

    def _calculate_distance(self, entity1: Dict, entity2: Dict) -> float:
        """计算实体间距离"""
        pos1 = entity1.get("position", [0, 0, 0])
        pos2 = entity2.get("position", [0, 0, 0])

        if isinstance(pos1, (list, tuple)) and len(pos1) >= 3 and \
                isinstance(pos2, (list, tuple)) and len(pos2) >= 3:
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            dz = pos2[2] - pos1[2]
            return math.sqrt(dx * dx + dy * dy + dz * dz)

        return float('inf')

    def get_collision_statistics(self, time_window: float = 60.0) -> Dict[str, Any]:
        """获取碰撞统计"""
        now = datetime.now()
        window_start = now - timedelta(seconds=time_window)

        collisions_in_window = [
            c for c in self.collision_history
            if c["timestamp"] >= window_start
        ]

        stats = {
            "total_collisions": len(self.collision_history),
            "collisions_in_window": len(collisions_in_window),
            "collision_rate_per_minute": len(collisions_in_window) / (time_window / 60),
            "collision_types": defaultdict(int),
            "most_colliding_entities": defaultdict(int)
        }

        for collision in collisions_in_window:
            coll_type = collision["details"].get("type", CollisionType.NO_COLLISION)
            stats["collision_types"][coll_type.value] += 1

            entities = collision["details"].get("entities", ("unknown", "unknown"))
            stats["most_colliding_entities"][entities[0]] += 1
            stats["most_colliding_entities"][entities[1]] += 1

        # 转换为普通字典并排序
        stats["collision_types"] = dict(stats["collision_types"])
        stats["most_colliding_entities"] = dict(
            sorted(stats["most_colliding_entities"].items(),
                   key=lambda x: x[1], reverse=True)[:10]
        )

        return stats

    def visualize_collisions(self, collisions: List[Dict]) -> str:
        """可视化碰撞（返回文本描述）"""
        if not collisions:
            return "无碰撞检测"

        visualization = "碰撞检测结果:\n"
        visualization += "=" * 50 + "\n"

        for i, collision in enumerate(collisions, 1):
            coll_type = collision.get("type", CollisionType.NO_COLLISION)
            entities = collision.get("entities", ("未知", "未知"))
            penetration = collision.get("penetration_depth", 0)
            relative_speed = collision.get("relative_speed", 0)

            visualization += f"碰撞 #{i}:\n"
            visualization += f"  实体: {entities[0]} ↔ {entities[1]}\n"
            visualization += f"  类型: {coll_type.value}\n"
            visualization += f"  穿透深度: {penetration:.3f}m\n"
            visualization += f"  相对速度: {relative_speed:.2f}m/s\n"

            if coll_type != CollisionType.NO_COLLISION:
                normal = collision.get("contact_normal", (0, 0, 0))
                visualization += f"  碰撞法线: ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})\n"

            visualization += "-" * 30 + "\n"

        return visualization
