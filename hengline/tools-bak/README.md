# 剧本智能分析工具模块

本模块提供了基于LlamaIndex的剧本智能分析功能，支持自定义剧本语法解析、结构化知识库构建和高效检索，帮助智能体深度理解原始剧本内容。

## 功能概览

- **剧本解析**：自定义剧本语法解析器，支持场景、角色、对话、动作等元素提取
- **结构化知识库**：构建剧本知识图谱，支持场景、角色、对话的关联存储
- **智能检索**：基于向量相似性的高效检索，支持场景查询、角色查询和内容查询
- **统计分析**：提供剧本统计信息，如场景数量、角色数量、对话统计等
- **LlamaIndex集成**：无缝集成LlamaIndex的文档处理和检索功能

## 安装依赖

请确保安装了以下依赖：

```bash
pip install llama-index-core llama-index-embeddings-openai llama-index-embeddings-huggingface llama-index-embeddings-ollama
```

## 快速开始

### 1. 剧本解析

```python
from hengline.tool import ScriptParser

# 创建解析器
parser = ScriptParser()

# 解析剧本文本
script_content = """INT. COFFEE SHOP - DAY

MARK (30s, casual)
Hey, how's it going?

SARAH (late 20s, friendly)
Not bad, just finishing up some work.

They exchange a smile."""

script_data = parser.parse(script_content)

# 解析剧本文件
script_data = parser.parse_file("path/to/script.txt")

# 创建LlamaIndex文档
llama_documents = parser.create_documents(script_content)
```

### 2. 构建剧本知识库

```python
from hengline.tool import ScriptKnowledgeBase

# 创建知识库
knowledge_base = ScriptKnowledgeBase(
    storage_dir="./script_knowledge",
    embedding_model_type="huggingface",
    embedding_model_name="BAAI/bge-small-zh-v1.5"
)

# 添加剧本文本
knowledge_base.add_script_text(script_content)

# 添加剧本文件
knowledge_base.add_script_file("path/to/script.txt")

# 添加剧本目录
knowledge_base.add_script_directory("path/to/scripts", recursive=True)

# 保存知识库
knowledge_base.save()
```

### 3. 智能检索与查询

```python
# 获取检索器
retriever = knowledge_base.get_retriever(
    similarity_top_k=3,
    similarity_threshold=0.7
)

# 查询知识库
results = knowledge_base.query("咖啡店场景中发生了什么？")

# 查询特定场景
scene_info = knowledge_base.query_scene("INT. COFFEE SHOP - DAY")

# 查询角色信息
character_info = knowledge_base.query_character("MARK")

# 获取统计信息
stats = knowledge_base.get_statistics()
```

### 4. 使用剧本智能分析

```python
from hengline.tools.script_intelligence_tool import ScriptIntelligence

# 创建智能分析对象
script_ai = ScriptIntelligence(
    storage_dir="./script_intelligence",
    embedding_model_type="huggingface",
    embedding_model_name="BAAI/bge-small-zh-v1.5"
)

# 分析剧本文本
analysis_result = script_ai.analyze_script_text(script_content)

# 分析剧本文件
analysis_result = script_ai.analyze_script_file("path/to/script.txt")

# 搜索内容
search_results = script_ai.search("关于工作的对话", top_k=5)

# 获取角色关系网络
character_relationships = script_ai.get_character_relationships()
```

## 模块说明

### ScriptParser

负责解析剧本内容，提取结构化信息。

- **parse**：解析剧本文本
- **parse_file**：解析剧本文件
- **create_documents**：创建LlamaIndex文档
- **_is_character_line**：判断是否为角色行
- **_add_character**：添加角色信息

### ScriptKnowledgeBase

管理剧本结构化数据并与LlamaIndex集成。

- **add_script_text**：添加剧本文本到知识库
- **add_script_file**：添加剧本文件到知识库
- **add_script_directory**：添加剧本目录到知识库
- **get_retriever**：获取检索器
- **query**：查询知识库
- **query_scene**：查询特定场景
- **query_character**：查询角色信息
- **get_statistics**：获取统计信息
- **save**：保存知识库
- **load**：加载知识库

### ScriptIntelligence

剧本智能分析主类，整合解析、知识库和检索功能。

- **analyze_script_text**：分析剧本文本
- **analyze_script_file**：分析剧本文件
- **analyze_script_directory**：分析剧本目录
- **search**：搜索内容
- **get_character_relationships**：获取角色关系
- **export_knowledge_base**：导出知识库
- **import_knowledge_base**：导入知识库

### LlamaIndex工具

提供LlamaIndex基础功能支持。

- **get_embedding_model**：获取不同类型的嵌入模型
- **create_vector_store**：创建或加载向量存储索引
- **create_index_from_directory**：从目录创建索引

## 配置说明

### 嵌入模型配置

支持三种类型的嵌入模型：

1. **OpenAI**：需要设置OPENAI_API_KEY环境变量
2. **HuggingFace**：自动下载模型到本地
3. **Ollama**：需要本地运行Ollama服务

### 知识库配置

- 通过storage_dir参数指定存储目录
- 支持配置分块大小和重叠区域
- 支持设置相似度阈值过滤低质量结果

### 剧本解析配置

- 支持识别标准剧本格式（场景标题、角色名、对话、动作描述）
- 支持自定义角色名识别规则

## 示例用例

### 剧本分析与检索

```python
from hengline.tools.script_intelligence_tool import ScriptIntelligence

# 创建智能分析对象
script_ai = ScriptIntelligence(
    storage_dir="./movie_script_knowledge",
    embedding_model_type="huggingface",
    embedding_model_name="BAAI/bge-small-zh-v1.5"
)

# 分析多部剧本
script_ai.analyze_script_directory("./scripts/movies", recursive=True)

# 查询角色信息
character_info = script_ai.search("主角 MARK 的性格特点")
print(character_info)

# 查询特定场景
coffee_shop_scenes = script_ai.search("咖啡店场景", top_k=3)
for scene in coffee_shop_scenes:
    print(f"场景: {scene['scene']}")
    print(f"内容摘要: {scene['content'][:100]}...")
    print()

# 获取统计信息
stats = script_ai.knowledge_base.get_statistics()
print(f"总场景数: {stats['total_scenes']}")
print(f"总角色数: {stats['total_characters']}")
print(f"总对话数: {stats['total_dialogues']}")
```

## 剧本格式说明

本解析器支持标准剧本格式，识别以下元素：

1. **场景标题**：以INT.或EXT.开头，如"INT. COFFEE SHOP - DAY"
2. **角色名**：大写字母，可包含描述，如"MARK (30s, casual)"
3. **对话**：角色名下一行的文本内容
4. **动作描述**：其他非角色名、非对话的描述性文本

## 注意事项

1. 使用OpenAI嵌入模型需要设置API密钥
2. 首次使用HuggingFace模型会自动下载，可能需要一些时间
3. 大型剧本集的索引创建可能比较耗时
4. 建议为生产环境配置持久化存储，避免重复创建索引
5. 复杂的非标准剧本格式可能需要自定义解析规则

## 故障排除

- **剧本解析错误**：检查剧本格式是否符合标准，特别注意场景标题和角色名的格式
- **嵌入模型初始化失败**：检查网络连接和API密钥配置
- **检索结果质量低**：尝试调整similarity_threshold和similarity_top_k参数
- **内存占用过高**：对于大型剧本集，考虑使用更高效的嵌入模型或分批处理

## 版本要求

- Python 3.8+
- llama-index-core >= 0.13.6
- 其他依赖请参考requirements.txt

---

更多详细信息请参考LlamaIndex官方文档：https://docs.llamaindex.ai/