# -*- coding: utf-8 -*-
"""
@FileName: logger.py
@Description: 自定义日志模块，支持按天创建日志文件、日志文件大小限制、控制台彩色输出等功能
            自定义日志模块，按天创建日志文件，最大10MB
@Author: HengLine
@Time: 2025/08 - 2025/11
"""

import os
import sys
import logging
import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

# 导入自定义的控制台颜色处理模块
from video_shot_breakdown.utils.console_colors import colored_log_formatter_factory, init_console_colors, IS_WINDOWS, HAS_COLORAMA
from video_shot_breakdown.utils.path_utils import PathResolver

# 初始化控制台颜色支持
init_console_colors()

class DailyRotatingFileHandler(RotatingFileHandler):
    """按天和文件大小旋转的日志处理器"""
    def __init__(self, base_dir: str, base_filename: str, max_bytes: int = 10*1024*1024, backup_count: int = 30, max_days: int = 15):
        self.base_dir = base_dir
        self.base_filename = base_filename
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.max_days = max_days  # 日志文件保留天数
        self.current_date = datetime.date.today()
        
        # 确保日志目录存在
        os.makedirs(self.base_dir, exist_ok=True)
        
        # 初始文件名
        self.current_log_file = self._get_log_filename()
        
        super().__init__(self.current_log_file, maxBytes=self.max_bytes, backupCount=self.backup_count, encoding='utf-8')
        
        # 清理过期日志文件
        self._cleanup_old_logs()
    
    def _get_log_filename(self) -> str:
        """获取当前日期的日志文件名"""
        today = datetime.date.today()
        date_str = today.strftime('%Y-%m-%d')
        
        # 基础文件名格式：base_dir/base_filename_YYYY-MM-DD.log
        base_log_file = os.path.join(self.base_dir, f"{self.base_filename}_{date_str}.log")
        
        # 检查是否需要创建序号后缀的文件
        count = 1
        log_file = base_log_file
        while os.path.exists(log_file) and os.path.getsize(log_file) >= self.max_bytes:
            log_file = os.path.join(self.base_dir, f"{self.base_filename}_{date_str}_{count}.log")
            count += 1
        
        return log_file
    
    def emit(self, record):
        """重写emit方法，实现按天和大小旋转"""
        # 检查日期是否变更
        today = datetime.date.today()
        if today != self.current_date:
            self.current_date = today
            self.current_log_file = self._get_log_filename()
            self.baseFilename = self.current_log_file
            
            # 关闭当前文件并打开新文件
            self.stream.close()
            self.mode = 'a'
            self.stream = self._open()
        
        # 检查文件大小是否超过限制
        if os.path.exists(self.current_log_file) and os.path.getsize(self.current_log_file) >= self.max_bytes:
            self.current_log_file = self._get_log_filename()
            self.baseFilename = self.current_log_file
            
            # 关闭当前文件并打开新文件
            self.stream.close()
            self.mode = 'a'
            self.stream = self._open()
        
        super().emit(record)
        
    def _cleanup_old_logs(self):
        """清理过期的日志文件，删除超过max_days天的日志"""
        try:
            # 获取当前日期
            today = datetime.date.today()
            
            # 计算过期日期
            expiration_date = today - datetime.timedelta(days=self.max_days)
            
            # 遍历日志目录中的所有文件
            if not os.path.exists(self.base_dir):
                return
            
            for filename in os.listdir(self.base_dir):
                file_path = os.path.join(self.base_dir, filename)
                
                # 只处理文件，不处理目录
                if not os.path.isfile(file_path):
                    continue
                
                # 检查文件名是否匹配日志文件格式
                # 格式: {base_filename}_YYYY-MM-DD.log 或 {base_filename}_YYYY-MM-DD_{n}.log
                try:
                    # 尝试解析日期部分
                    if filename.startswith(f"{self.base_filename}_") and filename.endswith(".log"):
                        # 提取文件名中间部分（去掉前缀和后缀）
                        middle_part = filename[len(f"{self.base_filename}_"):-4]  # -4 是去掉 ".log"
                        
                        # 尝试解析日期
                        # 处理格式: YYYY-MM-DD
                        if len(middle_part) == 10 and middle_part[4] == '-' and middle_part[7] == '-':
                            log_date = datetime.datetime.strptime(middle_part, '%Y-%m-%d').date()
                        # 处理格式: YYYY-MM-DD_{n}
                        elif '_' in middle_part:
                            date_part = middle_part.split('_')[0]
                            if len(date_part) == 10 and date_part[4] == '-' and date_part[7] == '-':
                                log_date = datetime.datetime.strptime(date_part, '%Y-%m-%d').date()
                            else:
                                continue  # 不符合日期格式，跳过
                        else:
                            continue  # 不符合日期格式，跳过
                        
                        # 检查是否过期
                        if log_date < expiration_date:
                            os.remove(file_path)
                except Exception:
                    # 如果解析失败，跳过该文件
                    continue
        except Exception as e:
            # 如果清理过程中出错，记录错误但不中断程序
            print(f"清理过期日志文件时出错: {str(e)}")

class Logger:
    """自定义日志类"""
    def __init__(self, name: str = 'HengLine', log_dir: Optional[str] = None, max_bytes: int = 10*1024*1024):
        """
        初始化日志器
        
        Args:
            name: 日志器名称
            log_dir: 日志目录路径，默认在项目根目录下的logs目录
            max_bytes: 单个日志文件最大字节数，默认10MB
        """
        # 初始化日志器
        self.logger = logging.getLogger(name)
        
        # 默认日志级别设为DEBUG
        self.logger.setLevel(logging.DEBUG)
        
        # 在Windows平台上，如果没有colorama库，记录一条警告
        if IS_WINDOWS and not HAS_COLORAMA:
            # 使用基本的print函数输出警告，因为日志系统尚未完全初始化
            print("警告: 在Windows平台上运行，但未安装colorama库，可能无法显示彩色日志。建议安装: pip install colorama")
        
        # 清除已有的处理器，避免重复添加
        if self.logger.handlers:
            self.logger.handlers.clear()

        # 获取日志配置
        try:
            # 设置日志级别，默认为DEBUG
            level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL
            }

            # 获取配置的日志级别
            log_level = level_map.get("INFO", logging.DEBUG)
            self.logger.setLevel(log_level)
            
        except ImportError:
            # 如果导入失败，使用默认的DEBUG级别
            pass
        
        # 如果已经有处理器，则清除
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 定义日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)  # 默认使用DEBUG级别
        
        # 使用带颜色的格式化器
        console_formatter = colored_log_formatter_factory('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
        
        try:
            # 获取配置以设置处理器级别

            
            level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL
            }
            
            # 获取配置的日志级别
            log_level = level_map.get("INFO", logging.INFO)
            
            # 设置控制台处理器级别
            console_level = level_map.get("INFO", log_level)
            console_handler.setLevel(console_level)
            
        except ImportError:
            # 如果导入失败，保持默认的DEBUG级别
            pass
        
        # 文件处理器
        if log_dir is None:
            # 默认日志目录：项目根目录下的logs文件夹
            # 使用当前文件所在路径向上回溯到项目根目录
            log_dir = os.path.join(PathResolver.get_project_root(), 'logs')
        
        file_handler = DailyRotatingFileHandler(log_dir, name, max_bytes, backup_count=30, max_days=15)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)  # 默认使用INFO级别
        self.logger.addHandler(file_handler)
        
        try:
            # 设置文件处理器级别
            level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL
            }
            
            # 获取配置的日志级别
            log_level = level_map.get("INFO", logging.INFO)
            
            # 设置文件处理器级别
            file_level = level_map.get("INFO", log_level)
            file_handler.setLevel(file_level)
            
            # 禁用不必要的日志
            if {}.get('disable_unnecessary_logs', True):
                # 禁用第三方库的日志
                for logger_name in ['urllib3', 'requests', 'PIL', 'matplotlib']:
                    logging.getLogger(logger_name).setLevel(logging.WARNING)
        except ImportError:
            # 如果导入失败，保持默认设置
            pass
    
    def debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """记录一般信息"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告信息"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误信息"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """记录严重错误信息"""
        self.logger.critical(message)

# 创建全局日志实例
logger = Logger(name="hengline")

# 方便使用的函数

def debug(message: str):
    logger.debug(message)

def info(message: str):
    logger.info(message)

def warning(message: str):
    logger.warning(message)

def error(message: str):
    logger.error(message)

def critical(message: str):
    logger.critical(message)

def log_with_context(level: str, message: str, context: dict = None) -> None:
    """
    记录带上下文的日志
    
    Args:
        level: 日志级别
        message: 日志消息
        context: 上下文信息
    """
    if context:
        # 将上下文信息格式化
        context_str = " ".join([f"{k}={v}" for k, v in context.items()])
        full_message = f"{message} | {context_str}"
    else:
        full_message = message
    
    # 根据级别记录日志
    if level == "DEBUG":
        debug(full_message)
    elif level == "INFO":
        info(full_message)
    elif level == "WARNING":
        warning(full_message)
    elif level == "ERROR":
        error(full_message)
    elif level == "CRITICAL":
        critical(full_message)

def log_function_call(func_name: str, params: dict = None, result = None) -> None:
    """
    记录函数调用信息
    
    Args:
        func_name: 函数名称
        params: 函数参数
        result: 函数返回结果
    """
    context = {"function": func_name}
    if params:
        context.update({f"param_{k}": str(v)[:50] if len(str(v)) > 50 else str(v) for k, v in params.items()})
    
    if result is not None:
        result_str = str(result)[:100] if len(str(result)) > 100 else str(result)
        context["result"] = result_str
    
    log_with_context("DEBUG", "Function call", context)

def log_performance(action: str, duration_ms: float, details: dict = None) -> None:
    """
    记录性能信息
    
    Args:
        action: 操作名称
        duration_ms: 耗时（毫秒）
        details: 详细信息
    """
    context = {"action": action, "duration_ms": duration_ms}
    if details:
        context.update(details)
    
    log_with_context("INFO", "Performance metric", context)