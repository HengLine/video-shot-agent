"""
@FileName: spatial_Index_analyzer.py
@Description: 空间索引 - 用于空间查询和空间关系管理
@Author: HengLine
@Time: 2026/1/4 22:18
"""
import math
from collections import defaultdict
from datetime import datetime
from typing import Tuple, List, Dict, Set, Optional, Any

from hengline.agent.continuity_guardian.validator.bounding_validator import BoundingVolumeType


class SpatialIndex:
    """空间索引 - 用于空间查询和空间关系管理"""

    def __init__(self, cell_size: float = 2.0):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int, int], Set[str]] = defaultdict(set)  # 网格单元格到实体ID的映射
        self.positions: Dict[str, Tuple[float, float, float]] = {}  # 实体ID到位置的映射
        self.bounding_volumes: Dict[str, Dict] = {}  # 实体边界体
        self.spatial_relationships: Dict[str, List[Dict]] = defaultdict(list)  # 实体间空间关系

    def update_position(self, entity_id: str, x: float, y: float, z: float = 0):
        """更新实体位置"""
        old_cell = None
        old_position = self.positions.get(entity_id)

        if old_position is not None:
            old_cell = self._get_grid_cell(old_position[0], old_position[1], old_position[2])

        # 更新位置
        self.positions[entity_id] = (x, y, z)
        new_cell = self._get_grid_cell(x, y, z)

        # 如果单元格改变，更新网格
        if old_cell != new_cell:
            if old_cell is not None and entity_id in self.grid[old_cell]:
                self.grid[old_cell].remove(entity_id)
                # 如果单元格为空，删除它
                if not self.grid[old_cell]:
                    del self.grid[old_cell]

            self.grid[new_cell].add(entity_id)

        # 更新空间关系
        self._update_spatial_relationships(entity_id, (x, y, z))

    def _get_grid_cell(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """获取坐标对应的网格单元格"""
        return (
            int(x / self.cell_size),
            int(y / self.cell_size),
            int(z / self.cell_size)
        )

    def _update_spatial_relationships(self, entity_id: str, position: Tuple[float, float, float]):
        """更新空间关系"""
        # 查找邻近实体
        neighbors = self.find_nearby_entities(entity_id, radius=5.0)

        # 更新关系
        current_relationships = []

        for neighbor_id in neighbors:
            if neighbor_id == entity_id:
                continue

            neighbor_pos = self.positions.get(neighbor_id)
            if neighbor_pos is None:
                continue

            # 计算空间关系
            relationship = self._calculate_spatial_relationship(
                entity_id, position, neighbor_id, neighbor_pos
            )

            if relationship:
                current_relationships.append(relationship)

        # 更新关系存储
        self.spatial_relationships[entity_id] = current_relationships

    def _calculate_spatial_relationship(self, entity1_id: str, pos1: Tuple[float, float, float],
                                        entity2_id: str, pos2: Tuple[float, float, float]) -> Optional[Dict]:
        """计算空间关系"""
        # 计算距离
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance == 0:
            return None

        # 计算方向
        direction = self._calculate_direction(dx, dy, dz)

        # 确定关系类型
        relationship_type = self._determine_relationship_type(distance, direction)

        # 获取边界体信息
        bv1 = self.bounding_volumes.get(entity1_id)
        bv2 = self.bounding_volumes.get(entity2_id)

        # 计算重叠和接触
        overlap_info = self._calculate_overlap_info(pos1, bv1, pos2, bv2)

        relationship = {
            "entity1": entity1_id,
            "entity2": entity2_id,
            "distance": distance,
            "direction": direction,
            "type": relationship_type,
            "overlap": overlap_info,
            "timestamp": datetime.now()
        }

        return relationship

    def _calculate_direction(self, dx: float, dy: float, dz: float) -> Dict[str, float]:
        """计算方向"""
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance == 0:
            return {"x": 0, "y": 0, "z": 0}

        return {
            "x": dx / distance,
            "y": dy / distance,
            "z": dz / distance
        }

    def _determine_relationship_type(self, distance: float,
                                     direction: Dict[str, float]) -> str:
        """确定关系类型"""
        if distance < 0.1:
            return "touching"
        elif distance < 1.0:
            return "near"
        elif distance < 3.0:
            return "close"
        elif distance < 10.0:
            return "medium_range"
        else:
            return "far"

    def _calculate_overlap_info(self, pos1: Tuple[float, float, float], bv1: Optional[Dict],
                                pos2: Tuple[float, float, float], bv2: Optional[Dict]) -> Dict[str, Any]:
        """计算重叠信息"""
        overlap = {
            "intersecting": False,
            "distance_between_surfaces": 0,
            "overlap_volume": 0
        }

        if bv1 is None or bv2 is None:
            return overlap

        # 简化的重叠检测
        bv_type1 = bv1.get("type", BoundingVolumeType.SPHERE)
        bv_type2 = bv2.get("type", BoundingVolumeType.SPHERE)

        if bv_type1 == BoundingVolumeType.SPHERE and bv_type2 == BoundingVolumeType.SPHERE:
            radius1 = bv1.get("radius", 0.5)
            radius2 = bv2.get("radius", 0.5)

            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            dz = pos2[2] - pos1[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            if distance < (radius1 + radius2):
                overlap["intersecting"] = True
                overlap["distance_between_surfaces"] = distance - (radius1 + radius2)

                # 计算重叠体积（两个球体重叠部分）
                r1, r2 = radius1, radius2
                if distance < abs(r1 - r2):
                    # 一个球完全在另一个内部
                    overlap["overlap_volume"] = (4 / 3) * math.pi * min(r1, r2) ** 3
                elif distance < (r1 + r2):
                    # 部分重叠
                    h1 = r1 - (distance ** 2 + r1 ** 2 - r2 ** 2) / (2 * distance)
                    h2 = r2 - (distance ** 2 + r2 ** 2 - r1 ** 2) / (2 * distance)

                    volume1 = (math.pi * h1 ** 2 * (3 * r1 - h1)) / 3
                    volume2 = (math.pi * h2 ** 2 * (3 * r2 - h2)) / 3
                    overlap["overlap_volume"] = volume1 + volume2

        return overlap

    def set_bounding_volume(self, entity_id: str, bv_type: BoundingVolumeType,
                            **kwargs):
        """设置实体边界体"""
        bv_data = {"type": bv_type}
        bv_data.update(kwargs)
        self.bounding_volumes[entity_id] = bv_data

    def find_nearby_entities(self, entity_id: str, radius: float) -> List[str]:
        """查找附近的实体"""
        if entity_id not in self.positions:
            return []

        position = self.positions[entity_id]
        center_cell = self._get_grid_cell(position[0], position[1], position[2])

        # 计算需要检查的单元格范围
        cells_to_check = self._get_cells_in_radius(center_cell, radius)

        nearby_entities = set()

        # 检查每个单元格
        for cell in cells_to_check:
            if cell in self.grid:
                for other_id in self.grid[cell]:
                    if other_id == entity_id:
                        continue

                    other_pos = self.positions.get(other_id)
                    if other_pos is None:
                        continue

                    # 计算实际距离
                    dx = other_pos[0] - position[0]
                    dy = other_pos[1] - position[1]
                    dz = other_pos[2] - position[2]
                    distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                    if distance <= radius:
                        nearby_entities.add(other_id)

        return list(nearby_entities)

    def _get_cells_in_radius(self, center_cell: Tuple[int, int, int],
                             radius: float) -> List[Tuple[int, int, int]]:
        """获取半径内的所有单元格"""
        cells = []

        # 计算单元格半径
        cell_radius = int(math.ceil(radius / self.cell_size))

        cx, cy, cz = center_cell

        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                for dz in range(-cell_radius, cell_radius + 1):
                    cells.append((cx + dx, cy + dy, cz + dz))

        return cells

    def ray_cast(self, origin: Tuple[float, float, float],
                 direction: Tuple[float, float, float],
                 max_distance: float = 100.0) -> List[Dict[str, Any]]:
        """射线投射"""
        hits = []

        # 简化实现：检查所有实体
        for entity_id, position in self.positions.items():
            bv = self.bounding_volumes.get(entity_id)
            if bv is None:
                continue

            # 简化的射线-边界体相交测试
            hit_info = self._test_ray_bv_intersection(origin, direction,
                                                      position, bv, max_distance)

            if hit_info["hit"]:
                hit_info["entity_id"] = entity_id
                hits.append(hit_info)

        # 按距离排序
        hits.sort(key=lambda x: x["distance"])

        return hits

    def _test_ray_bv_intersection(self, origin: Tuple[float, float, float],
                                  direction: Tuple[float, float, float],
                                  position: Tuple[float, float, float],
                                  bv: Dict, max_distance: float) -> Dict[str, Any]:
        """测试射线与边界体相交"""
        bv_type = bv.get("type", BoundingVolumeType.SPHERE)

        if bv_type == BoundingVolumeType.SPHERE:
            return self._test_ray_sphere_intersection(origin, direction,
                                                      position, bv, max_distance)
        elif bv_type == BoundingVolumeType.BOX:
            return self._test_ray_box_intersection(origin, direction,
                                                   position, bv, max_distance)
        else:
            return {"hit": False, "distance": 0, "point": (0, 0, 0)}

    def _test_ray_sphere_intersection(self, origin: Tuple[float, float, float],
                                      direction: Tuple[float, float, float],
                                      center: Tuple[float, float, float],
                                      bv: Dict, max_distance: float) -> Dict[str, Any]:
        """测试射线与球体相交"""
        radius = bv.get("radius", 0.5)

        # 射线参数方程: P = origin + t * direction
        # 球体方程: |P - center|² = radius²

        oc = (origin[0] - center[0],
              origin[1] - center[1],
              origin[2] - center[2])

        a = direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2
        b = 2 * (oc[0] * direction[0] + oc[1] * direction[1] + oc[2] * direction[2])
        c = oc[0] ** 2 + oc[1] ** 2 + oc[2] ** 2 - radius ** 2

        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return {"hit": False, "distance": 0, "point": (0, 0, 0)}

        sqrt_disc = math.sqrt(discriminant)

        # 求根
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        # 找到在范围内的最近正根
        t = None
        if 0 <= t1 <= max_distance:
            t = t1
        if 0 <= t2 <= max_distance:
            if t is None or t2 < t:
                t = t2

        if t is None:
            return {"hit": False, "distance": 0, "point": (0, 0, 0)}

        # 计算交点
        hit_point = (
            origin[0] + t * direction[0],
            origin[1] + t * direction[1],
            origin[2] + t * direction[2]
        )

        return {
            "hit": True,
            "distance": t,
            "point": hit_point,
            "normal": self._calculate_direction(
                hit_point[0] - center[0],
                hit_point[1] - center[1],
                hit_point[2] - center[2]
            )
        }

    def _test_ray_box_intersection(self, origin: Tuple[float, float, float],
                                   direction: Tuple[float, float, float],
                                   position: Tuple[float, float, float],
                                   bv: Dict, max_distance: float) -> Dict[str, Any]:
        """测试射线与盒体相交"""
        # 简化的轴对齐包围盒测试
        size = bv.get("size", (1.0, 1.0, 1.0))
        half_size = (size[0] / 2, size[1] / 2, size[2] / 2)

        min_bound = (
            position[0] - half_size[0],
            position[1] - half_size[1],
            position[2] - half_size[2]
        )
        max_bound = (
            position[0] + half_size[0],
            position[1] + half_size[1],
            position[2] + half_size[2]
        )

        # 计算射线与每个平面的交点
        t_min = 0.0
        t_max = max_distance

        for i in range(3):
            if abs(direction[i]) < 1e-6:
                # 射线与平面平行
                if origin[i] < min_bound[i] or origin[i] > max_bound[i]:
                    return {"hit": False, "distance": 0, "point": (0, 0, 0)}
            else:
                t1 = (min_bound[i] - origin[i]) / direction[i]
                t2 = (max_bound[i] - origin[i]) / direction[i]

                if t1 > t2:
                    t1, t2 = t2, t1

                t_min = max(t_min, t1)
                t_max = min(t_max, t2)

                if t_min > t_max:
                    return {"hit": False, "distance": 0, "point": (0, 0, 0)}

        if t_min > max_distance:
            return {"hit": False, "distance": 0, "point": (0, 0, 0)}

        # 计算交点
        hit_point = (
            origin[0] + t_min * direction[0],
            origin[1] + t_min * direction[1],
            origin[2] + t_min * direction[2]
        )

        # 计算法线（哪个面被击中）
        normal = (0, 0, 0)
        epsilon = 1e-6

        for i in range(3):
            if abs(hit_point[i] - min_bound[i]) < epsilon:
                normal = tuple(1 if j == i else 0 for j in range(3))
                break
            elif abs(hit_point[i] - max_bound[i]) < epsilon:
                normal = tuple(-1 if j == i else 0 for j in range(3))
                break

        return {
            "hit": True,
            "distance": t_min,
            "point": hit_point,
            "normal": normal
        }

    def get_spatial_relationships(self, entity_id: str) -> List[Dict]:
        """获取实体的空间关系"""
        return self.spatial_relationships.get(entity_id, [])

    def get_all_relationships(self) -> Dict[str, List[Dict]]:
        """获取所有空间关系"""
        return dict(self.spatial_relationships)

    def get_spatial_statistics(self) -> Dict[str, Any]:
        """获取空间统计信息"""
        stats = {
            "total_entities": len(self.positions),
            "entities_with_bv": len(self.bounding_volumes),
            "grid_cells_used": len(self.grid),
            "average_entities_per_cell": len(self.positions) / max(1, len(self.grid)),
            "spatial_density": 0,
            "relationship_count": sum(len(rels) for rels in self.spatial_relationships.values())
        }

        # 计算空间密度
        if self.positions:
            # 估算边界框
            all_x = [pos[0] for pos in self.positions.values()]
            all_y = [pos[1] for pos in self.positions.values()]
            all_z = [pos[2] for pos in self.positions.values()]

            width = max(all_x) - min(all_x) if all_x else 0
            height = max(all_y) - min(all_y) if all_y else 0
            depth = max(all_z) - min(all_z) if all_z else 0

            volume = width * height * depth
            if volume > 0:
                stats["spatial_density"] = len(self.positions) / volume

        return stats

    def visualize_spatial_distribution(self) -> str:
        """可视化空间分布（文本）"""
        if not self.positions:
            return "无实体数据"

        # 创建简单的2D网格可视化（忽略Z轴）
        all_x = [pos[0] for pos in self.positions.values()]
        all_y = [pos[1] for pos in self.positions.values()]

        if not all_x or not all_y:
            return "位置数据不足"

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # 创建网格
        grid_size = 20
        grid = [['·' for _ in range(grid_size)] for _ in range(grid_size)]

        for pos in self.positions.values():
            # 归一化到网格坐标
            x_norm = (pos[0] - min_x) / (max_x - min_x) if max_x > min_x else 0.5
            y_norm = (pos[1] - min_y) / (max_y - min_y) if max_y > min_y else 0.5

            grid_x = int(x_norm * (grid_size - 1))
            grid_y = int(y_norm * (grid_size - 1))

            # 确保在网格范围内
            grid_x = max(0, min(grid_size - 1, grid_x))
            grid_y = max(0, min(grid_size - 1, grid_y))

            # 标记位置（使用字母表示不同实体）
            entity_index = list(self.positions.keys()).index(list(self.positions.keys())[0])
            marker = chr(ord('A') + (entity_index % 26))
            grid[grid_y][grid_x] = marker

        # 创建可视化字符串
        visualization = "空间分布（2D投影）:\n"
        visualization += f"X范围: {min_x:.1f} 到 {max_x:.1f}\n"
        visualization += f"Y范围: {min_y:.1f} 到 {max_y:.1f}\n"
        visualization += "=" * (grid_size + 2) + "\n"

        for row in grid:
            visualization += "|" + "".join(row) + "|\n"

        visualization += "=" * (grid_size + 2) + "\n"
        visualization += f"实体总数: {len(self.positions)}\n"

        return visualization
