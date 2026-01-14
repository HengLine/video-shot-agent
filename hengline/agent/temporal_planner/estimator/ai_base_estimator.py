"""
@FileName: ai_estimator.py
@Description: AI估算器基类
@Author: HengLine
@Time: 2026/1/12 21:03
"""
import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any

from hengline.agent.temporal_planner.base_temporal_planner import EstimationError, EstimationErrorLevel
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, ElementType
from hengline.prompts.temporal_planner_prompt import PromptConfig
from hengline.prompts.temporal_planner_specialized_prompt import SpecializedPromptTemplates


class BaseAIDurationEstimator(ABC):
    """AI估算器基类"""
    def __init__(self, llm_client, config: PromptConfig = None):
        self.llm = llm_client
        self.config = config or PromptConfig()
        self.prompt_templates = SpecializedPromptTemplates(config)

        self.error_log: List[EstimationError] = []
        self.cache: Dict[str, DurationEstimation] = {}

        self.error_log: List[EstimationError] = []
        self.cache: Dict[str, DurationEstimation] = {}

    # ============================ 抽象属性（子类必须实现）============================
    @abstractmethod
    def _get_element_type(self) -> ElementType:
        """获取元素类型"""
        pass

    @abstractmethod
    def _get_id_value(self, element_data: Any) -> str:
        """获取元素ID字段名"""
        pass

    @abstractmethod
    def _parse_ai_response(self, response: str, element_data: Any) -> Dict[str, Any] | None:
        """解析AI响应（子类必须实现）"""
        pass

    @abstractmethod
    def _validate_estimation(self, parsed_result: Dict, element_data: Any) -> DurationEstimation:
        """验证估算结果（子类必须实现）"""
        pass

    @abstractmethod
    def _create_fallback_estimation(self, element_data: Any, context: Dict = None) -> DurationEstimation:
        """创建降级估算（子类必须实现）"""
        pass

    @abstractmethod
    def _generate_prompt(self, element_data: Any, context: Dict = None) -> str:
        """生成提示词（子类可重写）"""
        # 基础提示词由子类实现
        raise NotImplementedError("子类必须实现 _generate_prompt 方法")


    # ============================ 公共方法 ============================
    def estimate(self, element_data: Any, context: Dict = None) -> DurationEstimation:
        """公共接口：估算元素时长"""
        return self.estimate_with_context(element_data, context)


    def estimate_with_context(self, element_data: Any, context: Dict = None) -> DurationEstimation:
        """带上下文的估算"""
        start_time = datetime.now()

        try:
            # 检查缓存
            cache_key = self._generate_cache_key(element_data, context)
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                cached.timestamp = datetime.now().isoformat()  # 更新时间戳
                return cached

            # 生成提示词
            prompt = self._generate_prompt(element_data, context)
            prompt_hash = self._hash_prompt(prompt)

            # 调用AI
            raw_response = self._call_llm_with_retry(prompt)

            # 解析响应
            parsed_result = self._parse_ai_response(raw_response, element_data)

            # 验证和增强
            result = self._validate_estimation(parsed_result, element_data)
            result = self._enhance_estimation(result, element_data)

            # 添加元数据
            result.prompt_hash = prompt_hash
            result.timestamp = datetime.now().isoformat()
            result.processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            result.original_data = element_data.copy()
            result.raw_ai_response = raw_response

            # 缓存结果
            self.cache[cache_key] = result

            return result

        except Exception as e:
            return self._handle_estimation_error(element_data, context, str(e), start_time)

    def batch_estimate(self, elements: List[Dict], context: Dict = None) -> List[DurationEstimation]:
        """批量估算"""
        results = []

        for element_data in elements:
            result = self.estimate_with_context(element_data, context)
            results.append(result)

            # 更新上下文供下一个元素使用
            if context is not None:
                context = self._update_context(context, result)

        return results

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> Any | None:
        """调用LLM，支持重试"""
        for attempt in range(max_retries):
            try:
                return self.llm.invoke(prompt)

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"LLM调用失败: {e}")
                time.sleep(1)

    def _enhance_estimation(self, result: DurationEstimation, element_data: Any) -> DurationEstimation:
        """增强估算结果（子类可重写）"""
        # 基础增强：添加时间戳
        if not result.timestamp:
            result.timestamp = datetime.now().isoformat()
        return result

    def _update_context(self, context: Dict, result: DurationEstimation) -> Dict:
        """更新上下文（子类可重写）"""
        if "processed_elements" not in context:
            context["processed_elements"] = []

        context["processed_elements"].append({
            "element_id": result.element_id,
            "element_type": result.element_type.value,
            "duration": result.estimated_duration,
            "confidence": result.confidence
        })

        return context

    def _generate_cache_key(self, element_data: Any, context: Dict = None) -> str:
        """生成缓存键"""
        element_id = self._get_id_value(element_data)
        element_type = self._get_element_type().value

        # 使用元素内容和上下文生成哈希
        content_str = json.dumps(element_data, sort_keys=True)
        context_str = json.dumps(context, sort_keys=True) if context else ""

        combined = f"{element_type}:{element_id}:{content_str}:{context_str}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    def _hash_prompt(self, prompt: str) -> str:
        """计算提示词哈希"""
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]

    def _clean_json_response(self, response: str) -> str:
        """清理JSON响应"""
        # 移除Markdown代码块
        cleaned = re.sub(r'```json\s*', '', response)
        cleaned = re.sub(r'\s*```', '', cleaned)
        cleaned = cleaned.strip()

        # 提取JSON对象
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1

        if start >= 0 and end > start:
            cleaned = cleaned[start:end]

        return cleaned

    def _log_error(self, element_id: str, error_type: str, message: str,
                   level: EstimationErrorLevel, recovery_action: str = "",
                   fallback_value: float = None):
        """记录错误"""
        error = EstimationError(
            element_id=element_id,
            error_type=error_type,
            message=message,
            level=level,
            recovery_action=recovery_action,
            fallback_value=fallback_value,
            timestamp=datetime.now().isoformat()
        )

        self.error_log.append(error)

        # 打印错误信息
        level_icon = {
            EstimationErrorLevel.WARNING: "⚠️",
            EstimationErrorLevel.ERROR: "❌",
            EstimationErrorLevel.CRITICAL: "🔥"
        }.get(level, "ℹ️")

        print(f"{level_icon} [{level.value.upper()}] {error_type}: {message}")
        if recovery_action:
            print(f"  恢复操作: {recovery_action}")

    def _handle_estimation_error(self, element_data: Any, context: Dict,
                                 error_message: str, start_time: datetime) -> DurationEstimation:
        """处理估算错误"""
        self._log_error(
            element_id=self._get_id_value(element_data),
            error_type="estimation_failed",
            message=f"{self._get_element_type().value}估算失败: {error_message}",
            level=EstimationErrorLevel.ERROR,
            recovery_action="使用降级估算",
            fallback_value=0.0
        )

        # 创建降级估算
        fallback_result = self._create_fallback_estimation(element_data, context)

        # 添加错误元数据
        fallback_result.timestamp = datetime.now().isoformat()
        fallback_result.processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        fallback_result.confidence = min(fallback_result.confidence, 0.4)  # 降低置信度

        return fallback_result

    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        error_counts = {}
        for error in self.error_log:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1

        return {
            "total_errors": len(self.error_log),
            "error_by_type": error_counts,
            "errors_by_level": {
                "warning": len([e for e in self.error_log if e.level == EstimationErrorLevel.WARNING]),
                "error": len([e for e in self.error_log if e.level == EstimationErrorLevel.ERROR]),
                "critical": len([e for e in self.error_log if e.level == EstimationErrorLevel.CRITICAL])
            }
        }

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()

    def clear_errors(self):
        """清空错误日志"""
        self.error_log.clear()
