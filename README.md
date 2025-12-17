# 剧本分镜智能体

一个基于多智能体协作的剧本分镜系统，能够将自然语言剧本拆分为AI可生成的短视频脚本单元，输出高质量分镜片段描述，并保证叙事连续性。支持多种AI提供商，具有强大的可扩展性和易用性。可以通过Python库、Web API、LangGraph节点或A2A系统集成使用。

> - **需求描述**：假如我有一段预估两分钟左右的剧本，想通过AI模型生成对应的短视频。
>
> - **技术受限**：目前的各种模型仅支持一次生成5-10秒长度的视频，想要生成两分钟长度的视频，只能通过“拼接”的方式，将多个5秒的片段合成为一个视频。
>
> - **任务&挑战点**：要实现视频拼接，第一步就需要拆分原剧本，拆分后的剧本尽量接近5-10秒时长（取决于模型），且每个视频片段还必须要保持连贯性，不然生成的视频片段合成后会导致场景、动作、人物等衔接不上。
>
>   且剧情中的动作、语速等会影响时长，所以需要考虑多种情景，比如：老人动作慢、生气怒吼时语速会较快、跑比走要快等等。
>
>   这便是本智能体需要完成的任务，用户只需要给出剧本，而后根据各种技术拆解，最后将拆解完成的剧本片段返回，用户只需要将其交给模型（Runway、Pika、Sora、Wan、Stable Video等）生成即可，最后再利用相关技术将片段合成为完整视频。

**视频创作流程**：客户端  → LLM 剧本创作  →  <u>***剧本解析（剧本拆分）***</u> → DM 视频生成（文生视频） →  视频合成渲染（FFmpeg）

**注意**：本智能体不会参与剧本创作，不会调用模型生成视频，亦不会合成视频，以上流程中标注处就是本智能体的任务。


## 核心功能

- **智能剧本解析**：自动识别场景、对话和动作指令，理解故事结构
- **精准时序规划**：按短视频粒度智能切分内容，分配合理时长
- **连续性守护**：确保相邻分镜间角色状态、场景和情节的一致性
- **高质量分镜生成**：生成详细的中文画面描述和英文AI视频提示词
- **多模型支持**：兼容OpenAI、Qwen、DeepSeek、Ollama等多种AI提供商

详细设计参照文档：[**剧本分镜智能体的架构设计与实现细节**](https://pengline.github.io/2025/10/0194020a663c408fb500dd7532349519/)

## 快速上手

### 1. 环境准备

**前置条件**：Python 3.10 或更高版本

```bash
# 克隆项目
git clone https://github.com/HengLine/video-shot-agent.git
cd video-shot-agent

######### 方式1：自动安装
# 脚本会自动创建虚拟环境、安装依赖并启动服务，若失败，可手动安装
python start_app.py


######### 方式2：手动安装
python -m venv .venv
# 激活虚拟环境 (Windows)
.venv\Scripts\activate
# 或者 (Linux/Mac)
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

复制配置文件并设置环境变量：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置必要的参数：

```properties
# 选择AI提供商：openai, qwen, deepseek, ollama
AI_PROVIDER=qwen

# 根据选择的提供商配置对应的API密钥
QWEN_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxx
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus

# 嵌入模型配置，支持AI供应商：ollama、huggingface、openai
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=quentinz/bge-large-zh-v1.5:latest
```

### 3. 启动应用

```bash
python start_app.py
```

应用将在 `http://0.0.0.0:8000` 启动，提供API接口服务。

## 使用方法

### 1. 作为Python库使用

```python
from hengline.generate_agent import generate_storyboard

# 基本使用：传入中文剧本文本
script_text = """
深夜11点，城市公寓客厅，窗外大雨滂沱。
林然裹着旧羊毛毯蜷在沙发里，电视静音播放着黑白老电影。
茶几上半杯凉茶已凝出水雾，旁边摊开一本旧相册。
手机突然震动，屏幕亮起"未知号码"。
她盯着看了三秒，指尖悬停...
"""

# 生成分镜
result = generate_storyboard(script_text)
print(f"生成了 {result['total_shots']} 个分镜")
for shot in result['shots']:
    print(f"\n分镜 {shot['shot_id']}:")
    print(f"时间: {shot['start_time']}-{shot['end_time']}s")
    print(f"描述: {shot['description']}")
    print(f"AI提示词: {shot['ai_prompt']}")
```

### 2. 集成到Web应用（API）

可以通过HTTP API将剧本分镜智能体集成到各种Web应用中：

```python
from flask import Flask, request, jsonify
from hengline.generate_agent import generate_storyboard

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    result = generate_storyboard(
        script_text=data['script_text'],
        style=data.get('style', 'realistic')
    )
    return jsonify(result)
```

API接口调用

```bash
curl -X POST http://localhost:8000/api/generate_storyboard \
  -H "Content-Type: application/json" \
  -d '{"script_text": "深夜11点，城市公寓客厅...", "style": "realistic", "duration_per_shot": 5}'
```

> - `script_text`：中文剧本文本（必填）
> - `style`：分镜风格，可选值：`realistic`, `anime`, `cinematic`, `cartoon`（默认：`realistic`）
> - `duration_per_shot`：每段分镜目标时长（秒），默认：`5`

### 3. 集成到LangGraph节点

可以将剧本分镜智能体作为LangGraph工作流中的一个节点：

```python
from langgraph.graph import Graph, StateGraph, END
from hengline.generate_agent import generate_storyboard

# 定义工作流状态
class StoryWorkflowState:
    def __init__(self):
        self.script = ""
        self.storyboard = None
        self.status = "pending"

# 创建分镜生成节点
def generate_storyboard_node(state):
    try:
        state.storyboard = generate_storyboard(
            script_text=state.script,
            style="cinematic",
            enable_quality_check=True
        )
        state.status = "completed"
    except Exception as e:
        state.status = f"error: {str(e)}"
    return state

# 构建LangGraph工作流
graph = StateGraph(StoryWorkflowState)
graph.add_node("storyboard_generator", generate_storyboard_node)
graph.set_entry_point("storyboard_generator")
graph.add_edge("storyboard_generator", END)

# 编译并运行工作流
app = graph.compile()

# 执行工作流
result = app.invoke({
    "script": "深夜11点，城市公寓客厅，窗外大雨滂沱...",
    "status": "pending"
})
```

### 4. 集成到A2A系统

将剧本分镜智能体集成到Agent-to-Agent协作系统中：

```python
from hengline.agent.multi_agent_pipeline import MultiAgentPipeline
from hengline.agent.workflow_nodes import StoryboardAgentNode

# 创建多智能体协作管道
pipeline = MultiAgentPipeline()

# 添加分镜智能体节点
storyboard_node = StoryboardAgentNode(
    name="storyboard_generator",
    params={
        "style": "realistic",
        "duration_per_shot": 5,
        "enable_continuity_check": True
    }
)

# 配置工作流
pipeline.add_node(storyboard_node)

# 设置输入输出连接
pipeline.connect(
    source="script_provider",  # 上游提供剧本的节点
    target="storyboard_generator",
    input_mapping={"script": "script_text"}
)

pipeline.connect(
    source="storyboard_generator",
    target="video_renderer",  # 下游视频渲染节点
    output_mapping={"shots": "storyboard_frames"}
)

# 执行协作流程
result = pipeline.run({
    "script_provider": {"script": "深夜11点，城市公寓客厅..."}
})
```



## 输入输出示例

输入：中文剧本

```python
script_text = """
深夜11点，城市公寓客厅，窗外大雨滂沱。
林然裹着旧羊毛毯蜷在沙发里，电视静音播放着黑白老电影。
茶几上半杯凉茶已凝出水雾，旁边摊开一本旧相册。
手机突然震动，屏幕亮起"未知号码"。
她盯着看了三秒，指尖悬停...
"""

result = generate_storyboard(script_text, style="cinematic")
```

输出：结构化分镜结果

```json
{
  "total_shots": 3,
  "storyboard_title": "深夜来电",
  "status": "success",
  "shots": [
    {
      "shot_id": "shot_001",
      "start_time": 0.0,
      "end_time": 5.0,
      "duration": 5.0,
      "description": "城市公寓客厅，深夜场景。窗外大雨猛烈敲击玻璃...",
      "ai_prompt": "A dimly lit city apartment living room at midnight...",
      "characters": ["林然"],
      "dialogue": "",
      "camera_angle": "medium shot",
      "continuity_anchors": ["林然: 坐姿, 沙发", "环境: 客厅, 雨夜"]
    }
    // 更多分镜...
  ],
  "final_continuity_state": {...},
  "warnings": []
}
```

## 故障排除

1. **API连接问题**：检查API密钥和网络连接
2. **分镜生成失败**：尝试简化剧本描述，增加重试次数
3. **连续性问题**：确保相邻场景描述包含足够的上下文信息

## 联系方式

如有问题或建议，请提交GitHub Issue或联系项目维护团队。