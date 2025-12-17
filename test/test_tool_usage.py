#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
剧本智能分析工具使用示例
展示如何使用剧本解析器、知识库和智能检索功能
"""

import json
import os

# 嵌入模型获取
from hengline.client.embedding_client import get_embedding_client
# 导入工具模块
from hengline.tools import (
    # 剧本智能分析主类
    create_script_intelligence,
    # 剧本知识库
    create_script_knowledge_base,
)
from hengline.tools.script_parser_tool import ScriptParserTool


def example_basic_script_parsing():
    """
    示例：基本的剧本解析功能
    """
    print("\n=== 基本剧本解析示例 ===")

    # 示例剧本文本
    sample_script = """
    INT. COFFEE SHOP - DAY
    
    A cozy coffee shop with soft jazz playing in the background. SARAH, a young professional in her 20s,
    sits at a table, typing on her laptop. She looks up as JAMES, a friendly barista, approaches.
    
    JAMES
    (smiling)
    Your usual latte?
    
    SARAH
    (nodding)
    Yes, please. And could I get an extra shot today?
    
    JAMES
    No problem. Long day?
    
    SARAH
    You have no idea. Deadline is tomorrow.
    
    EXT. CITY PARK - EVENING
    
    The sun sets over the park. MARK, Sarah's friend, waits on a bench. Sarah approaches, still holding her coffee cup.
    
    MARK
    There you are! I was starting to worry.
    
    SARAH
    Sorry, work ran late. You know how it is.
    
    MARK
    I get it. Let's walk and talk.
    
    They start walking through the park.
    FADE OUT.
    """

    # 创建解析器并解析
    parser = ScriptParserTool()
    parsed_result = parser.parse(sample_script)

    # 显示解析结果
    print(f"解析完成！发现{parsed_result['stats']['scene_count']}个场景，{parsed_result['stats']['character_count']}个角色")
    print("\n场景列表:")
    for scene in parsed_result['scenes']:
        print(f"- {scene['heading']} (角色: {', '.join(scene['characters'])})")

    print("\n角色列表:")
    for character_name, character_info in parsed_result['characters'].items():
        print(f"- {character_name}: {character_info['dialogue_count']} 次对话")


def example_script_knowledge_base():
    """
    示例：使用剧本知识库
    """
    print("\n=== 剧本知识库使用示例 ===")

    # 创建临时存储目录
    storage_dir = "./temp_knowledge_base"
    os.makedirs(storage_dir, exist_ok=True)

    try:
        # 获取嵌入模型（使用默认的OpenAI模型）
        print("正在初始化嵌入模型...")
        embedding_model = get_embedding_client(model_type="openai")

        # 创建知识库
        kb = create_script_knowledge_base(
            embedding_model=embedding_model,
            storage_dir=storage_dir
        )

        # 示例剧本
        sample_script = """
        INT. APARTMENT - MORNING
        
        ALICE, a 30-year-old architect, stands at the window, sipping coffee. She looks thoughtful.
        
        ALICE
        (to herself)
        Today's the day. I can do this.
        
        She takes a deep breath and picks up her portfolio.
        
        CUT TO:
        
        INT. DESIGN OFFICE - DAY
        
        A modern office space. BOB, the hiring manager, sits behind a desk. Alice enters, nervous but determined.
        
        BOB
        Please have a seat, Ms. Johnson.
        
        ALICE
        Thank you, Mr. Thompson.
        
        BOB
        I've reviewed your work. Impressive. Tell me about your design philosophy.
        """

        # 添加剧本到知识库
        print("添加剧本到知识库...")
        result = kb.add_script_text(sample_script, "sample_script_001")
        print(f"添加结果: {result['status']}, 场景数: {result['scene_count']}")

        # 创建检索器并查询
        print("\n创建检索器并执行查询...")
        kb.create_retriever(search_type="similarity", similarity_top_k=3)

        # 查询示例
        queries = [
            "Alice在面试中说了什么",
            "描述办公室场景",
            "Bob的角色"
        ]

        for query in queries:
            print(f"\n查询: '{query}'")
            results = kb.query(query)
            for i, result in enumerate(results['results'], 1):
                print(f"结果 {i}: {result['text'][:100]}...")

        # 获取统计信息
        stats = kb.get_statistics()
        print(f"\n知识库统计: {stats['document_count']}个文档, {stats['scene_count']}个场景")

    finally:
        # 清理临时文件（实际使用时可能需要保留）
        import shutil
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir)


def example_script_intelligence():
    """
    示例：使用剧本智能分析主类
    """
    print("\n=== 剧本智能分析示例 ===")

    # 创建智能分析实例
    intelligence = create_script_intelligence(
        embedding_model_name="openai",
        storage_dir="./temp_intelligence"
    )

    try:
        # 示例剧本文件内容（实际使用时可以读取真实文件）
        sample_script = """
        INT. LIBRARY - QUIET
        
        EMMA, a graduate student, sits at a table surrounded by books. She's deeply focused on her laptop.
        DAVID, her classmate, approaches quietly.
        
        DAVID
        (whispering)
        Hey, how's the research going?
        
        EMMA
        (startled)
        David! You scared me. It's going... slowly.
        
        DAVID
        Need any help? I just finished my part.
        
        EMMA
        Actually, I could use a fresh pair of eyes on this chapter.
        
        They start discussing the research papers.
        
        INT. CAFETERIA - LATER
        
        Emma and David sit at a table, eating lunch. They continue their discussion.
        
        EMMA
        I think we should focus on the environmental impact section.
        
        DAVID
        Agreed. That's where the strongest evidence is.
        """

        # 分析剧本
        print("分析剧本文本...")
        analysis_result = intelligence.analyze_script_text(sample_script)

        # 显示分析结果
        print("\n分析报告:")
        print(f"- 总场景数: {analysis_result['analysis']['overview']['total_scenes']}")
        print(f"- 总角色数: {analysis_result['analysis']['overview']['total_characters']}")
        print(f"- 平均场景行数: {analysis_result['analysis']['overview']['avg_scene_lines']}")

        # 显示角色分析
        print("\n角色分析:")
        for char in analysis_result['analysis']['character_analysis']['top_characters_by_dialogue']:
            print(f"- {char['name']}: {char['dialogue_count']} 次对话")

        # 执行智能搜索
        print("\n智能搜索示例:")
        search_queries = [
            "图书馆场景中的对话",
            "Emma和David讨论了什么",
            "研究相关的内容"
        ]

        for query in search_queries:
            print(f"\n搜索: '{query}'")
            search_results = intelligence.search(query, top_k=2)
            for result in search_results['results']:
                print(f"- 类型: {result['result_type']}, 相关性: {result['relevance']}")
                print(f"  内容: {result['text'][:80]}...")

        # 获取特定场景信息
        scene_info = intelligence.get_scene_info(1)
        if scene_info:
            print(f"\n场景1信息: {scene_info['heading']}")

    finally:
        # 清理临时存储
        intelligence.clear_all()


def example_file_analysis():
    """
    示例：从文件分析剧本（需要有实际文件）
    """
    print("\n=== 剧本文件分析示例 ===")
    print("注意：此示例需要提供实际的剧本文件路径")

    # 这里仅展示API使用方式，实际运行需要替换为真实文件路径
    example_file_path = "./example_script.txt"

    if os.path.exists(example_file_path):
        # 创建智能分析实例
        intelligence = create_script_intelligence()

        # 分析文件
        print(f"分析文件: {example_file_path}")
        result = intelligence.analyze_script_file(example_file_path)

        # 显示结果
        print(f"分析完成，发现{result['analysis']['overview']['total_scenes']}个场景")

        # 保存分析结果
        with open("./script_analysis_result.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print("分析结果已保存到 script_analysis_result.json")
    else:
        print(f"示例文件不存在: {example_file_path}")
        print("请创建一个剧本文件并修改此示例中的文件路径")
        print("\n示例文件内容格式:")
        print("""
        INT. LOCATION - TIME
        
        DESCRIPTION
        
        CHARACTER
        Dialogue text.
        """)


def main():
    """
    运行所有示例
    """
    print("\n===== 剧本智能分析工具使用示例 =====")

    try:
        # 运行基本解析示例
        example_basic_script_parsing()

        # 运行知识库示例
        example_script_knowledge_base()

        # 运行智能分析示例
        example_script_intelligence()

        # 运行文件分析示例（仅演示API）
        example_file_analysis()

    except Exception as e:
        print(f"\n示例运行出错: {str(e)}")
        print("注意：某些示例可能需要配置API密钥或安装特定依赖")

    print("\n===== 示例结束 =====")
    print("\n使用提示:")
    print("1. 确保已安装所有必要的依赖（pip install -r requirements.txt）")
    print("2. 使用OpenAI嵌入模型时，需要设置OPENAI_API_KEY环境变量")
    print("3. 可以替换示例中的剧本内容为您自己的剧本")
    print("4. 对于生产环境，建议配置持久化的存储目录")


if __name__ == "__main__":
    main()
