#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain 工具集成示例
展示如何将 HengLine 的剧本分析工具转换为 LangChain Tool 格式
"""

import json
from typing import Dict, Any, Optional, List

from langchain.agents import AgentType, initialize_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.tools import Tool, tool

from hengline.client.client_factory import ClientFactory
# 导入 HengLine 工具
from hengline.tools import (
    create_script_intelligence
)
from hengline.tools.script_parser_tool import ScriptParserTool


# =========================================
# 方法1: 使用 @tool 装饰器创建 LangChain Tool
# =========================================

@tool
def parse_script(script_text: str) -> Dict[str, Any]:
    """
    解析剧本文本，提取场景、角色、对话等信息
    
    Args:
        script_text: 剧本文本内容
        
    Returns:
        解析结果，包含场景、角色等信息
    """
    parser = ScriptParserTool()
    return parser.parse(script_text)


@tool
def analyze_script_content(script_text: str, script_id: Optional[str] = None) -> Dict[str, Any]:
    """
    智能分析剧本文本，包括解析、统计和特征提取
    
    Args:
        script_text: 剧本文本内容
        script_id: 可选的剧本ID
        
    Returns:
        分析结果，包含统计信息和特征
    """
    # 使用默认配置创建剧本智能分析实例
    script_intel = create_script_intelligence(
        embedding_model_name="openai",  # 可以根据需要修改为其他模型
        storage_dir=None  # 内存模式运行
    )
    return script_intel.analyze_script_text(script_text, script_id)


@tool
def search_script_knowledge(query: str, storage_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    在剧本知识库中搜索相关内容
    
    Args:
        query: 搜索查询
        storage_dir: 知识库存储目录
        
    Returns:
        搜索结果
    """
    script_intel = create_script_intelligence(
        embedding_model_name="openai",
        storage_dir=storage_dir
    )
    return script_intel.search(query)


# =========================================
# 方法2: 手动创建 Tool 对象
# =========================================

def create_script_parser_tool() -> Tool:
    """
    创建剧本解析器工具
    """

    def _parse_script(script_text: str) -> str:
        parser = ScriptParserTool()
        result = parser.parse(script_text)
        return json.dumps(result, ensure_ascii=False, indent=2)

    return Tool(
        name="ScriptParser",
        func=_parse_script,
        description="用于解析剧本文本，提取场景、角色、对话等结构化信息。输入为剧本文本字符串。"
    )


def create_script_analyzer_tool() -> Tool:
    """
    创建剧本分析器工具
    """

    def _analyze_script(script_text: str) -> str:
        script_intel = create_script_intelligence(
            embedding_model_name="openai",
            storage_dir=None
        )
        result = script_intel.analyze_script_text(script_text)
        return json.dumps(result, ensure_ascii=False, indent=2)

    return Tool(
        name="ScriptAnalyzer",
        func=_analyze_script,
        description="用于智能分析剧本，包括解析、统计特征提取和深度理解。输入为剧本文本字符串。"
    )


# =========================================
# 方法3: 创建自定义工具类
# =========================================

class LangChainScriptTool:
    """
    LangChain 剧本工具包装类
    """

    def __init__(self):
        self.parser = ScriptParserTool()
        self.script_intel = None

    def initialize_intelligence(self, embedding_model_name: str = "openai",
                                storage_dir: Optional[str] = None):
        """
        初始化剧本智能分析组件
        """
        self.script_intel = create_script_intelligence(
            embedding_model_name=embedding_model_name,
            storage_dir=storage_dir
        )

    def get_tools(self) -> List[Tool]:
        """
        获取所有可用工具
        """
        tools = []

        # 添加剧本解析工具
        tools.append(Tool(
            name="ParseScript",
            func=self._parse_script_wrapper,
            description="解析剧本文本，提取场景、角色、对话等元素"
        ))

        # 添加剧本分析工具
        tools.append(Tool(
            name="AnalyzeScript",
            func=self._analyze_script_wrapper,
            description="智能分析剧本内容，包括统计信息和特征提取"
        ))

        # 添加知识库搜索工具
        tools.append(Tool(
            name="SearchScriptKnowledge",
            func=self._search_knowledge_wrapper,
            description="在剧本知识库中搜索相关内容"
        ))

        return tools

    def _parse_script_wrapper(self, script_text: str) -> str:
        """包装剧本解析函数"""
        result = self.parser.parse(script_text)
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _analyze_script_wrapper(self, script_text: str) -> str:
        """包装剧本分析函数"""
        if not self.script_intel:
            self.initialize_intelligence()
        result = self.script_intel.analyze_script_text(script_text)
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _search_knowledge_wrapper(self, query: str) -> str:
        """包装知识库搜索函数"""
        if not self.script_intel:
            self.initialize_intelligence()
        result = self.script_intel.search(query)
        return json.dumps(result, ensure_ascii=False, indent=2)


# =========================================
# 使用示例
# =========================================

def example_with_decorated_tools():
    """
    使用装饰器定义的工具示例
    """
    print("\n=== 使用装饰器定义的 LangChain 工具示例 ===")

    # 示例剧本
    sample_script = """
    INT. COFFEE SHOP - DAY
    
    一个舒适的咖啡店，背景音乐播放着柔和的爵士乐。SARAH，一位20多岁的年轻职业女性，
    坐在桌旁，正在笔记本电脑上打字。当JAMES，一位友好的咖啡师，走近时，她抬起头。
    
    JAMES
    (微笑着)
    您的常规拿铁？
    
    SARAH
    (点头)
    是的，请。今天能多加一份浓缩吗？
    
    JAMES
    没问题。漫长的一天？
    
    SARAH
    你无法想象。明天截止。
    """

    # 直接使用装饰器定义的工具
    print("\n1. 使用 @tool 装饰的 parse_script 工具:")
    result = parse_script(sample_script)
    print(f"场景数量: {len(result['scenes'])}")
    print(f"角色数量: {len(result['characters'])}")

    print("\n2. 使用 @tool 装饰的 analyze_script_content 工具:")
    analysis = analyze_script_content(sample_script)
    print(f"分析完成，包含统计信息: {list(analysis.keys())}")


def example_with_manual_tools():
    """
    使用手动创建的工具示例
    """
    print("\n=== 使用手动创建的 LangChain 工具示例 ===")

    # 示例剧本
    sample_script = """
    EXT. PARK - MORNING
    
    阳光明媚的公园，孩子们在玩耍，老人们在散步。LI，一位中年男子，
    坐在长椅上看书。WANG，他的朋友，走过来。
    
    WANG
    嘿，老李！今天这么早就出来了？
    
    LI
    是啊，早起的鸟儿有虫吃。
    """

    # 创建工具
    parser_tool = create_script_parser_tool()
    analyzer_tool = create_script_analyzer_tool()

    # 使用工具
    print("\n1. 使用手动创建的 ScriptParser 工具:")
    parser_result = parser_tool.run(sample_script)
    print("解析结果摘要:")
    # 解析JSON结果以显示关键信息
    parsed_json = json.loads(parser_result)
    print(f"场景标题: {parsed_json['scenes'][0]['heading']}")
    print(f"出场角色: {', '.join(parsed_json['scenes'][0]['characters'])}")


def example_with_agent():
    """
    将工具集成到 LangChain Agent 示例
    """
    print("\n=== LangChain Agent 集成示例 ===")

    try:
        # 获取LangChain LLM实例
        llm = ClientFactory.get_langchain_llm(provider="openai")

        if not llm:
            print("警告: 无法获取LangChain LLM实例，将使用模拟模式")
            return

        # 创建自定义工具类实例
        script_tool = LangChainScriptTool()
        script_tool.initialize_intelligence()

        # 获取工具列表
        tools = script_tool.get_tools()

        # 初始化Agent
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        # 示例查询
        query = "分析这段剧本，告诉我有多少个场景和角色：\n\n"
        query += """
        INT. OFFICE - DAY
        
        一个繁忙的办公室。ZHANG坐在办公桌前，正在打电话。
        LIU走进来，手里拿着文件。
        
        LIU
        张总，这是您要的报告。
        
        ZHANG
        谢谢，先放这儿吧。
        """

        print("\n使用Agent分析剧本:")
        # 注意：实际运行需要有效的API密钥
        # result = agent.run(query)
        # print(result)
        print("提示: 要运行完整的Agent示例，请确保配置了有效的API密钥")

    except Exception as e:
        print(f"运行Agent示例时出错: {str(e)}")
        print("请确保已安装所有依赖并配置了正确的API密钥")


def example_with_chain():
    """
    将工具与LangChain Chain结合使用示例
    """
    print("\n=== LangChain Chain 结合工具示例 ===")

    try:
        # 创建剧本解析工具
        parser_tool = create_script_parser_tool()

        # 创建提示模板
        prompt = PromptTemplate(
            input_variables=["script_analysis", "question"],
            template="基于以下剧本分析结果，请回答问题：\n\n分析结果: {script_analysis}\n\n问题: {question}"
        )

        # 获取LLM
        llm = ClientFactory.get_langchain_llm(provider="openai")

        if not llm:
            print("警告: 无法获取LangChain LLM实例，将使用简化模式")
            # 使用模拟回答
            sample_script = """
            INT. CLASSROOM - AFTERNOON
            
            教室里有学生们在讨论。TEACHER站在讲台上。
            
            STUDENT 1
            老师，这个问题我不太明白。
            
            TEACHER
            没关系，让我再解释一遍。
            """

            # 使用工具解析剧本
            analysis = parser_tool.run(sample_script)
            print(f"\n剧本解析完成，场景信息: {json.loads(analysis)['scenes'][0]['heading']}")
            return

        # 创建Chain
        chain = LLMChain(llm=llm, prompt=prompt)

        # 示例剧本
        sample_script = """
        INT. CLASSROOM - AFTERNOON
        
        教室里有学生们在讨论。TEACHER站在讲台上。
        
        STUDENT 1
        老师，这个问题我不太明白。
        
        TEACHER
        没关系，让我再解释一遍。
        """

        # 先使用工具解析剧本
        analysis = parser_tool.run(sample_script)

        # 然后使用Chain回答问题
        question = "这个场景中有哪些角色？"
        # 注意：实际运行需要有效的API密钥
        # result = chain.run(script_analysis=analysis, question=question)
        # print(f"问题: {question}")
        # print(f"回答: {result}")
        print("提示: 要运行完整的Chain示例，请确保配置了有效的API密钥")

    except Exception as e:
        print(f"运行Chain示例时出错: {str(e)}")


def main():
    """
    运行所有示例
    """
    print("===== LangChain 工具集成示例 =====")

    try:
        # 运行装饰器工具示例
        example_with_decorated_tools()

        # 运行手动工具示例
        example_with_manual_tools()

        # 运行Agent示例
        example_with_agent()

        # 运行Chain示例
        example_with_chain()

    except Exception as e:
        print(f"示例运行出错: {str(e)}")

    print("\n===== 示例结束 =====")
    print("\n使用提示:")
    print("1. 确保已安装所有必要的依赖")
    print("2. 运行完整功能时，需要配置相应的API密钥")
    print("3. 可以根据需要修改示例中的剧本内容")
    print("4. 对于生产环境，建议配置持久化的存储目录")


if __name__ == "__main__":
    main()
