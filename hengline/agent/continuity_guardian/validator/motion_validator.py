"""
@FileName: motion_validator.py
@Description: 运动分析器
@Author: HengLine
@Time: 2026/1/4 21:57
"""
import math
from enum import Enum
from typing import List, Tuple, Dict, Any

import numpy as np

from hengline.agent.continuity_guardian.validator.bounding_validator import TrajectoryType

class MotionType(Enum):
    """运动类型枚举"""
    STATIC = "static"          # 静止
    LINEAR = "linear"          # 线性运动
    CURVED = "curved"          # 曲线运动
    ACCELERATED = "accelerated" # 加速运动
    OSCILLATING = "oscillating" # 振荡运动
    RANDOM = "random"          # 随机运动


class MotionAnalyzer:
    """运动分析器"""

    def __init__(self, sampling_rate: float = 0.1):
        self.sampling_rate = sampling_rate
        self.motion_patterns: Dict[str, Dict] = {}
        self.trajectory_cache: Dict[str, List] = {}

    def analyze_motion(self, trajectory: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        """分析运动轨迹"""
        if len(trajectory) < 2:
            return {"error": "轨迹点太少"}

        analysis = {
            "trajectory_type": self._classify_trajectory(trajectory),
            "total_distance": self._calculate_total_distance(trajectory),
            "average_speed": self._calculate_average_speed(trajectory),
            "acceleration_profile": self._calculate_acceleration_profile(trajectory),
            "smoothness_metrics": self._calculate_smoothness_metrics(trajectory),
            "curvature_analysis": self._analyze_curvature(trajectory),
            "energy_efficiency": self._calculate_energy_efficiency(trajectory),
            "physical_plausibility": self._assess_physical_plausibility(trajectory)
        }

        # 检测运动模式
        analysis["motion_patterns"] = self._detect_motion_patterns(trajectory)

        return analysis

    def _classify_trajectory(self, trajectory: List[Tuple]) -> TrajectoryType:
        """分类轨迹类型"""
        if len(trajectory) < 3:
            return TrajectoryType.LINEAR

        # 计算曲率和扭率
        curvatures = self._calculate_curvatures(trajectory)
        avg_curvature = np.mean(curvatures) if curvatures else 0

        # 检查线性度
        linear_score = self._calculate_linearity_score(trajectory)

        # 检查周期性
        periodic_score = self._calculate_periodicity_score(trajectory)

        if linear_score > 0.9:
            return TrajectoryType.LINEAR
        elif avg_curvature > 0.5 and periodic_score > 0.7:
            return TrajectoryType.CIRCULAR
        elif avg_curvature > 0.3:
            return TrajectoryType.PARABOLIC
        elif periodic_score > 0.6:
            return TrajectoryType.OSCILLATING
        else:
            return TrajectoryType.RANDOM_WALK

    def _calculate_total_distance(self, trajectory: List[Tuple]) -> float:
        """计算总移动距离"""
        total = 0.0
        for i in range(1, len(trajectory)):
            point1 = trajectory[i - 1]
            point2 = trajectory[i]
            if len(point1) >= 3 and len(point2) >= 3:
                dx = point2[0] - point1[0]
                dy = point2[1] - point1[1]
                dz = point2[2] - point1[2]
                total += math.sqrt(dx * dx + dy * dy + dz * dz)
        return total

    def _calculate_average_speed(self, trajectory: List[Tuple]) -> float:
        """计算平均速度"""
        if len(trajectory) < 2:
            return 0.0

        total_distance = self._calculate_total_distance(trajectory)
        total_time = (len(trajectory) - 1) * self.sampling_rate

        if total_time > 0:
            return total_distance / total_time
        return 0.0

    def _calculate_acceleration_profile(self, trajectory: List[Tuple]) -> Dict[str, Any]:
        """计算加速度分布"""
        if len(trajectory) < 3:
            return {"max_acceleration": 0, "average_acceleration": 0, "jerk": 0}

        accelerations = []
        jerks = []

        for i in range(2, len(trajectory)):
            # 计算速度
            v1 = self._calculate_velocity(trajectory[i - 2], trajectory[i - 1], self.sampling_rate)
            v2 = self._calculate_velocity(trajectory[i - 1], trajectory[i], self.sampling_rate)

            # 计算加速度
            acceleration = self._calculate_velocity(v1, v2, self.sampling_rate)
            accel_magnitude = math.sqrt(acceleration[0] ** 2 + acceleration[1] ** 2 + acceleration[2] ** 2)
            accelerations.append(accel_magnitude)

            # 如果有前一个加速度，计算急动度
            if i > 2:
                prev_accel = accelerations[-2]
                jerk = abs(accel_magnitude - prev_accel) / self.sampling_rate
                jerks.append(jerk)

        profile = {
            "max_acceleration": max(accelerations) if accelerations else 0,
            "average_acceleration": np.mean(accelerations) if accelerations else 0,
            "max_jerk": max(jerks) if jerks else 0,
            "average_jerk": np.mean(jerks) if jerks else 0,
            "acceleration_std": np.std(accelerations) if accelerations else 0
        }

        return profile

    def _calculate_smoothness_metrics(self, trajectory: List[Tuple]) -> Dict[str, float]:
        """计算平滑度指标"""
        if len(trajectory) < 3:
            return {"jerk_score": 0, "curvature_consistency": 0, "overall_smoothness": 0}

        # 计算急动度分数
        jerk_score = self._calculate_jerk_score(trajectory)

        # 计算曲率一致性
        curvature_consistency = self._calculate_curvature_consistency(trajectory)

        # 计算速度变化平滑度
        speed_smoothness = self._calculate_speed_smoothness(trajectory)

        overall = (jerk_score + curvature_consistency + speed_smoothness) / 3

        return {
            "jerk_score": jerk_score,
            "curvature_consistency": curvature_consistency,
            "speed_smoothness": speed_smoothness,
            "overall_smoothness": overall
        }

    def _analyze_curvature(self, trajectory: List[Tuple]) -> Dict[str, Any]:
        """分析曲率"""
        if len(trajectory) < 3:
            return {"average_curvature": 0, "max_curvature": 0, "curvature_profile": []}

        curvatures = self._calculate_curvatures(trajectory)

        if curvatures:
            return {
                "average_curvature": np.mean(curvatures),
                "max_curvature": max(curvatures),
                "min_curvature": min(curvatures),
                "curvature_std": np.std(curvatures),
                "curvature_profile": curvatures,
                "torsion": self._calculate_torsion(trajectory) if len(trajectory) >= 4 else 0
            }

        return {"average_curvature": 0, "max_curvature": 0, "curvature_profile": []}

    def _calculate_energy_efficiency(self, trajectory: List[Tuple]) -> Dict[str, float]:
        """计算能量效率"""
        if len(trajectory) < 2:
            return {"efficiency_score": 0, "energy_waste": 0, "optimality_ratio": 0}

        # 计算实际路径长度
        actual_distance = self._calculate_total_distance(trajectory)

        # 计算起点到终点的直线距离
        start = trajectory[0]
        end = trajectory[-1]
        if len(start) >= 3 and len(end) >= 3:
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dz = end[2] - start[2]
            direct_distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        else:
            direct_distance = actual_distance

        # 计算效率
        if direct_distance > 0:
            efficiency_score = direct_distance / actual_distance
        else:
            efficiency_score = 1.0 if actual_distance == 0 else 0.0

        # 计算能量浪费（加速/减速次数）
        accel_profile = self._calculate_acceleration_profile(trajectory)
        speed_changes = self._count_speed_changes(trajectory)

        energy_waste = min(1.0, speed_changes / (len(trajectory) - 1))

        return {
            "efficiency_score": efficiency_score,
            "energy_waste": energy_waste,
            "optimality_ratio": efficiency_score * (1 - energy_waste),
            "path_deviation": actual_distance - direct_distance
        }

    def _assess_physical_plausibility(self, trajectory: List[Tuple]) -> Dict[str, Any]:
        """评估物理合理性"""
        plausibility = {
            "speed_limits_violated": False,
            "acceleration_limits_violated": False,
            "unnatural_turns": False,
            "teleportation_detected": False,
            "overall_plausibility": 1.0
        }

        if len(trajectory) < 2:
            return plausibility

        # 检查速度限制（假设最大速度10m/s）
        max_speed = 10.0
        speeds = []
        for i in range(1, len(trajectory)):
            speed = self._calculate_speed_between_points(trajectory[i - 1], trajectory[i])
            speeds.append(speed)
            if speed > max_speed:
                plausibility["speed_limits_violated"] = True

        # 检查加速度限制（假设最大加速度30m/s²）
        max_acceleration = 30.0
        accel_profile = self._calculate_acceleration_profile(trajectory)
        if accel_profile["max_acceleration"] > max_acceleration:
            plausibility["acceleration_limits_violated"] = True

        # 检查不自然的转向
        curvature_analysis = self._analyze_curvature(trajectory)
        if curvature_analysis.get("max_curvature", 0) > 5.0:  # 高曲率
            plausibility["unnatural_turns"] = True

        # 检查瞬移（点之间距离过大）
        for i in range(1, len(trajectory)):
            distance = self._calculate_distance_between_points(trajectory[i - 1], trajectory[i])
            if distance > 5.0:  # 假设最大合理距离5m
                plausibility["teleportation_detected"] = True

        # 计算总体合理性分数
        penalty = 0.0
        if plausibility["speed_limits_violated"]:
            penalty += 0.3
        if plausibility["acceleration_limits_violated"]:
            penalty += 0.3
        if plausibility["unnatural_turns"]:
            penalty += 0.2
        if plausibility["teleportation_detected"]:
            penalty += 0.5

        plausibility["overall_plausibility"] = max(0.0, 1.0 - penalty)

        return plausibility

    def _detect_motion_patterns(self, trajectory: List[Tuple]) -> List[str]:
        """检测运动模式"""
        patterns = []

        if len(trajectory) < 3:
            return patterns

        # 检测直线运动
        if self._is_linear_motion(trajectory):
            patterns.append("linear_motion")

        # 检测圆周运动
        if self._is_circular_motion(trajectory):
            patterns.append("circular_motion")

        # 检测振荡运动
        if self._is_oscillating_motion(trajectory):
            patterns.append("oscillating_motion")

        # 检测静止
        if self._is_stationary(trajectory):
            patterns.append("stationary")

        # 检测加速/减速
        if self._has_acceleration_pattern(trajectory):
            patterns.append("accelerating")
        if self._has_deceleration_pattern(trajectory):
            patterns.append("decelerating")

        return patterns

    def _calculate_velocity(self, point1: Tuple, point2: Tuple, dt: float) -> Tuple[float, float, float]:
        """计算速度向量"""
        if len(point1) >= 3 and len(point2) >= 3 and dt > 0:
            return (
                (point2[0] - point1[0]) / dt,
                (point2[1] - point1[1]) / dt,
                (point2[2] - point1[2]) / dt
            )
        return (0, 0, 0)

    def _calculate_speed_between_points(self, point1: Tuple, point2: Tuple) -> float:
        """计算两点间的速度大小"""
        if len(point1) >= 3 and len(point2) >= 3:
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            dz = point2[2] - point1[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            return distance / self.sampling_rate
        return 0.0

    def _calculate_distance_between_points(self, point1: Tuple, point2: Tuple) -> float:
        """计算两点间距离"""
        if len(point1) >= 3 and len(point2) >= 3:
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            dz = point2[2] - point1[2]
            return math.sqrt(dx * dx + dy * dy + dz * dz)
        return 0.0

    def _calculate_curvatures(self, trajectory: List[Tuple]) -> List[float]:
        """计算曲率"""
        curvatures = []

        for i in range(1, len(trajectory) - 1):
            if len(trajectory[i - 1]) >= 3 and len(trajectory[i]) >= 3 and len(trajectory[i + 1]) >= 3:
                # 使用三点计算曲率
                p0 = trajectory[i - 1]
                p1 = trajectory[i]
                p2 = trajectory[i + 1]

                # 计算向量
                v1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
                v2 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])

                # 计算叉积
                cross = self._cross_product(v1, v2)
                cross_magnitude = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)

                # 计算向量长度
                v1_length = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
                v2_length = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)

                # 计算曲率
                if v1_length > 0 and v2_length > 0:
                    curvature = cross_magnitude / (v1_length * v2_length)
                    curvatures.append(curvature)

        return curvatures

    def _cross_product(self, v1: Tuple, v2: Tuple) -> Tuple[float, float, float]:
        """计算叉积"""
        return (
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        )

    def _calculate_torsion(self, trajectory: List[Tuple]) -> float:
        """计算扭率（需要至少4个点）"""
        if len(trajectory) < 4:
            return 0.0

        # 简化扭率计算
        return 0.0

    def _calculate_linearity_score(self, trajectory: List[Tuple]) -> float:
        """计算线性度分数"""
        if len(trajectory) < 3:
            return 1.0

        # 使用主成分分析（PCA）思想
        points = np.array(trajectory)
        if points.shape[0] < 3 or points.shape[1] < 3:
            return 1.0

        # 计算协方差矩阵
        centered = points - np.mean(points, axis=0)
        covariance = np.cov(centered, rowvar=False)

        # 计算特征值
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.sort(eigenvalues)[::-1]

        # 线性度 = 最大特征值 / 特征值和
        if np.sum(eigenvalues) > 0:
            return eigenvalues[0] / np.sum(eigenvalues)

        return 1.0

    def _calculate_periodicity_score(self, trajectory: List[Tuple]) -> float:
        """计算周期性分数"""
        if len(trajectory) < 6:
            return 0.0

        # 简化周期性检测
        # 检查位置序列是否重复
        return 0.0

    def _calculate_jerk_score(self, trajectory: List[Tuple]) -> float:
        """计算急动度分数（越高越平滑）"""
        if len(trajectory) < 3:
            return 1.0

        accel_profile = self._calculate_acceleration_profile(trajectory)
        max_jerk = accel_profile.get("max_jerk", 0)

        # 将急动度转换为平滑度分数
        if max_jerk > 50:
            return 0.0
        elif max_jerk > 20:
            return 0.3
        elif max_jerk > 10:
            return 0.6
        elif max_jerk > 5:
            return 0.8
        else:
            return 1.0

    def _calculate_curvature_consistency(self, trajectory: List[Tuple]) -> float:
        """计算曲率一致性"""
        curvature_analysis = self._analyze_curvature(trajectory)
        curvature_std = curvature_analysis.get("curvature_std", 0)

        # 将曲率标准差转换为一致性分数
        if curvature_std > 2.0:
            return 0.0
        elif curvature_std > 1.0:
            return 0.3
        elif curvature_std > 0.5:
            return 0.6
        elif curvature_std > 0.2:
            return 0.8
        else:
            return 1.0

    def _calculate_speed_smoothness(self, trajectory: List[Tuple]) -> float:
        """计算速度平滑度"""
        if len(trajectory) < 3:
            return 1.0

        speeds = []
        for i in range(1, len(trajectory)):
            speed = self._calculate_speed_between_points(trajectory[i - 1], trajectory[i])
            speeds.append(speed)

        # 计算速度变化
        speed_changes = []
        for i in range(1, len(speeds)):
            speed_changes.append(abs(speeds[i] - speeds[i - 1]))

        if not speed_changes:
            return 1.0

        avg_speed_change = np.mean(speed_changes)
        avg_speed = np.mean(speeds)

        if avg_speed > 0:
            relative_change = avg_speed_change / avg_speed
        else:
            relative_change = 0

        # 将相对变化转换为平滑度分数
        if relative_change > 0.5:
            return 0.0
        elif relative_change > 0.3:
            return 0.3
        elif relative_change > 0.2:
            return 0.6
        elif relative_change > 0.1:
            return 0.8
        else:
            return 1.0

    def _count_speed_changes(self, trajectory: List[Tuple]) -> int:
        """计算速度变化次数"""
        if len(trajectory) < 3:
            return 0

        speeds = []
        for i in range(1, len(trajectory)):
            speed = self._calculate_speed_between_points(trajectory[i - 1], trajectory[i])
            speeds.append(speed)

        changes = 0
        for i in range(1, len(speeds)):
            if abs(speeds[i] - speeds[i - 1]) > 0.1:  # 速度变化超过0.1m/s
                changes += 1

        return changes

    def _is_linear_motion(self, trajectory: List[Tuple]) -> bool:
        """检测是否直线运动"""
        linearity_score = self._calculate_linearity_score(trajectory)
        return linearity_score > 0.95

    def _is_circular_motion(self, trajectory: List[Tuple]) -> bool:
        """检测是否圆周运动"""
        if len(trajectory) < 10:
            return False

        curvature_analysis = self._analyze_curvature(trajectory)
        avg_curvature = curvature_analysis.get("average_curvature", 0)
        curvature_std = curvature_analysis.get("curvature_std", 0)

        # 圆周运动特征：曲率较高且一致
        return avg_curvature > 0.3 and curvature_std < 0.1

    def _is_oscillating_motion(self, trajectory: List[Tuple]) -> bool:
        """检测是否振荡运动"""
        if len(trajectory) < 6:
            return False

        # 检查位置变化方向
        direction_changes = 0
        for i in range(2, len(trajectory)):
            dx1 = trajectory[i - 1][0] - trajectory[i - 2][0]
            dx2 = trajectory[i][0] - trajectory[i - 1][0]

            if dx1 * dx2 < 0:  # 方向改变
                direction_changes += 1

        # 如果有足够多的方向改变，可能是振荡运动
        return direction_changes >= len(trajectory) // 3

    def _is_stationary(self, trajectory: List[Tuple]) -> bool:
        """检测是否静止"""
        if len(trajectory) < 2:
            return True

        total_distance = self._calculate_total_distance(trajectory)
        return total_distance < 0.01  # 总移动距离小于1cm

    def _has_acceleration_pattern(self, trajectory: List[Tuple]) -> bool:
        """检测是否有加速模式"""
        if len(trajectory) < 3:
            return False

        speeds = []
        for i in range(1, len(trajectory)):
            speed = self._calculate_speed_between_points(trajectory[i - 1], trajectory[i])
            speeds.append(speed)

        # 检查速度是否递增
        increasing_count = 0
        for i in range(1, len(speeds)):
            if speeds[i] > speeds[i - 1] + 0.1:  # 速度增加超过0.1m/s
                increasing_count += 1

        return increasing_count >= len(speeds) * 0.6  # 60%的时间在加速

    def _has_deceleration_pattern(self, trajectory: List[Tuple]) -> bool:
        """检测是否有减速模式"""
        if len(trajectory) < 3:
            return False

        speeds = []
        for i in range(1, len(trajectory)):
            speed = self._calculate_speed_between_points(trajectory[i - 1], trajectory[i])
            speeds.append(speed)

        # 检查速度是否递减
        decreasing_count = 0
        for i in range(1, len(speeds)):
            if speeds[i] < speeds[i - 1] - 0.1:  # 速度减少超过0.1m/s
                decreasing_count += 1

        return decreasing_count >= len(speeds) * 0.6  # 60%的时间在减速

    def predict_future_position(self, trajectory: List[Tuple],
                                steps: int = 10) -> List[Tuple[float, float, float]]:
        """预测未来位置"""
        if len(trajectory) < 3:
            return []

        # 使用简单的线性外推
        last_point = trajectory[-1]
        second_last = trajectory[-2]

        dx = last_point[0] - second_last[0]
        dy = last_point[1] - second_last[1]
        dz = last_point[2] - second_last[2]

        predictions = []
        for i in range(1, steps + 1):
            predictions.append((
                last_point[0] + dx * i,
                last_point[1] + dy * i,
                last_point[2] + dz * i
            ))

        return predictions

    def get_motion_summary(self, trajectory: List[Tuple[float, float, float]]) -> str:
        """获取运动摘要"""
        analysis = self.analyze_motion(trajectory)

        summary = f"运动分析摘要:\n"
        summary += f"轨迹类型: {analysis.get('trajectory_type', '未知').value}\n"
        summary += f"总距离: {analysis.get('total_distance', 0):.2f}米\n"
        summary += f"平均速度: {analysis.get('average_speed', 0):.2f}米/秒\n"

        accel = analysis.get('acceleration_profile', {})
        summary += f"最大加速度: {accel.get('max_acceleration', 0):.2f}米/秒²\n"

        smoothness = analysis.get('smoothness_metrics', {})
        summary += f"平滑度: {smoothness.get('overall_smoothness', 0):.2f}\n"

        plausibility = analysis.get('physical_plausibility', {})
        summary += f"物理合理性: {plausibility.get('overall_plausibility', 0):.2f}\n"

        patterns = analysis.get('motion_patterns', [])
        summary += f"检测到的模式: {', '.join(patterns) if patterns else '无'}\n"

        return summary