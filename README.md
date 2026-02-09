# 剧本分镜智能体

一个基于多智能体协作的剧本分镜系统，能够将多种格式的剧本拆分为AI可生成的短视频脚本单元，输出高质量分镜片段描述，并保证叙事连续性。支持多种AI提供商，具有强大的可扩展性和易用性。可以通过Python库、Web API、LangGraph节点或A2A系统集成使用。

> - **需求描述**：假如我有一段预估两分钟左右的剧本，想通过AI模型生成对应的短视频。
>
> - **技术受限**：目前的各种模型仅支持一次生成5-10秒长度的视频，想要生成两分钟长度的视频，只能通过“拼接”的方式，将多个5秒的片段合成为一个视频。
>
> - **任务&挑战点**：要实现视频拼接，第一步就需要拆分原剧本，拆分后的剧本尽量接近5-10秒时长（取决于模型），且每个视频片段还必须要保持连贯性，不然生成的视频片段合成后会导致场景、动作、人物等衔接不上。
>
>   且剧情中的动作、语速等会影响时长，所以需要考虑多种情景，比如：老人动作慢、生气怒吼时语速会较快、跑比走要快等等。
>
>   这便是本智能体需要完成的任务，用户只需要给出剧本，而后根据各种技术拆解，最后将拆解完成的剧本片段返回，用户只需要将其交给模型（Runway、Pika、Sora、Wan、Stable Video等）生成即可，最后再利用相关技术将片段合成为完整视频。

**视频创作流程**：客户端  → LLM 剧本创作  →  <u>***剧本解析（拆分）***</u> → DM 视频生成（文生视频） →  视频合成渲染（FFmpeg）

**注意**：本智能体不会参与剧本创作，目前版本不会调用模型生成视频，亦不会合成视频（未来版本会支持），以上流程中标注处就是本智能体的任务。


## 核心功能

- **智能剧本解析**：自动识别场景、对话和动作指令，理解故事结构
- **精准时序规划**：按镜头粒度智能切分内容，分配合理时长
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

# 安装为可编辑包
pip install -e .

######### 方式1：自动安装
# 脚本会自动创建虚拟环境、安装依赖并启动服务，若失败，可手动安装
python main.py


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
# 部署环境（development, production）
APP__ENVIRONMENT=development
# 剧本的语言设置，目前支持：zh（中文）或en（英文）
APP__LANGUAGE=zh
# ================= API配置 =================
#  服务器主机，支持HOST环境变量
API__HOST=localhost
#  服务器端口，支持PORT环境变量
API__PORT=8000

########################## LLM 模型配置 #########################
# 系统支持的厂商（openai, qwen, deepseek, ollama），当默认模型不可用时使用备用厂商

# ================= LLM默认配置 =================
# LLM 厂商 API
LLM__DEFAULT__BASE_URL=https://dashscope-intl.aliyuncs.com/api/v1
# LLM 厂商 KAY
LLM__DEFAULT__API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# LLM 厂商 模型
LLM__DEFAULT__MODEL_NAME=qwen-plus
# 生成温度参数，控制输出的随机性： 0.0 = 确定性输出，1.0 = 最大随机性
LLM__DEFAULT__TEMPERATURE=0.1
# 默认API超时时间（秒）
LLM__DEFAULT__TIMEOUT=60
# 最大重试次数
LLM__DEFAULT__MAX_RETRIES=2
# 最大生成令牌数
LLM__DEFAULT__MAX_TOKENS=3000
LLM__DEFAULT__RETRY_DELAY=1

# ================= LLM备用配置 =================
LLM__FALLBACK__BASE_URL=http://localhost:11434
LLM__FALLBACK__MODEL_NAME=qwen3:4b
LLM__FALLBACK__TEMPERATURE=0.1
LLM__FALLBACK__TIMEOUT=300
LLM__FALLBACK__MAX_TOKENS=5000


########################## 嵌入模型配置 #########################
# 系统支持的厂商（openai, qwen, HuggingFace, ollama），当默认模型不可用时使用备用厂商

# ================ 嵌入模型默认配置 ================
EMBED__DEFAULT__BASE_URL=https://api.openai.com/v1
EMBED__DEFAULT__API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# 中文常用模型：gte-large-zh，bge-large-zh-v1.5，text-embedding-3-small
EMBED__DEFAULT__MODEL_NAME=text-embedding-3-small
EMBED__DEFAULT__DEVICE=gpu
EMBED__DEFAULT__NORMALIZE_EMBEDDINGS=true
EMBED__DEFAULT__DIMENSIONS=1536
EMBED__DEFAULT__TIMEOUT=60
```

### 3. 启动应用

```bash
python main.py
```

应用将在 `http://0.0.0.0:8000` 启动，提供API接口服务。

### 4. 调用接口

提交任务：

```sh
curl --location --request POST 'http://localhost:8000/api/v1/storyboard' \
--header 'Content-Type: application/json' \
--data-raw '{
    "script": "剧本标题：《雨中的约定》\n时长：约30秒\n场景：城市街角咖啡店外，雨天\n角色：\n- 林小雨（女，20岁，学生，抱着一本湿漉漉的书）\n- 陈阳（男，22岁，兼职外卖员，穿着黄色雨衣）\n\n[开场]\n（雨声淅沥，镜头从灰蒙蒙的天空下摇，聚焦在咖啡店外的长椅上。林小雨蹲在长椅旁，用手帕擦拭一本被雨水浸湿的诗集，神情焦急。）\n林小雨（自言自语，带着哭腔）：\n\"明明说好今天还书的……这雨下得，他会不会不来了？\"\n\n[镜头切换]\n（陈阳骑着电动车冲进雨幕，后座外卖箱里露出一角蓝色封面的书。他刹车太急，差点撞上长椅，林小雨的书掉进水洼。）\n陈阳（手忙脚乱捡书，抬头）：\n\"对不起！这书……是你的？\"\n\n[特写]\n（两本书并排躺在水洼里——林小雨的《飞鸟集》，陈阳外卖箱里的同款书，封面上贴着\"借阅卡：陈阳→林小雨\"。）\n林小雨（愣住，突然笑了）：\n\"你迟到十分钟，但……书没湿透。\"\n\n陈阳（挠头，从雨衣里掏出干毛巾裹住书）：\n\"我跑了两条街找防水袋……诗里说'\''雨是云的眼泪'\''，可我不想让你哭。\"\n\n[结尾]\n（雨渐小，阳光穿透云层。林小雨翻开书，里面夹着一张电影票根，日期是下周三。陈阳脱下雨衣罩在她头上，两人并肩跑向屋檐，笑声渐远。）\n画外音（林小雨的旁白）：\n\"有些约定，会迟到，但永远不会缺席。\"\n\n[黑屏，字幕浮现]\n\"雨会停，而故事才刚刚开始。\"\n\n风格：清新治愈，带点幽默，适合短视频平台传播。\n核心冲突：用\"湿书\"和\"迟到\"制造小误会，通过\"同款书\"和\"电影票\"暗示双向暗恋，雨天象征情感转折。"
}'
```

获取任务结果：

```sh
# hengline202602061816441424 为任务提交成功后返回的 task_id
curl --location --request GET 'http://localhost:8000/api/v1/result/hengline202602061816441424'
```

查看任务状态：

```sh
# hengline202602061816441424 为任务提交成功后返回的 task_id
curl --location --request GET 'http://localhost:8000/api/v1/status/hengline202602061816441424'
```



## 嵌入智能体方式

### 1. 作为Python库使用

以下为同步方式

```python
from hengline.hengline_agent import generate_storyboard

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
print(f" 解析结果 {result['success']} ")
for shot in result['data']:
    print(f"\n分镜 {shot['fragments']}:")
    print(f"审查报告: {shot['audit_report']}")
    print(f"metadata: {shot['metadata']}")
```

### 2. 集成到Web应用（API）

可以通过 HTTP API 将剧本分镜智能体集成到各种 Web 应用中：

```python
from flask import Flask, request, jsonify
from hengline.generate_agent import generate_storyboard

app = Flask(__name__)

@app.route('/api/v1/generate', methods=['POST'])
def generate():
    data = request.json
    result = generate_storyboard(
        script_text=data['script_text'],
    )
    return jsonify(result)
```

获取任务结果

```python

```

API接口调用

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"script_text": "深夜11点，城市公寓客厅..."}'
```

> - `script_text`：中文剧本文本（必填）
> - `max_fragment_duration`：每段分镜目标最大时长（秒），默认：`5`

### 3. 集成到LangGraph节点

可以将剧本分镜智能体作为 LangGraph 工作流中的一个节点：

```python
from langgraph.graph import Graph, StateGraph, END
from hengline.hengline_agent import generate_storyboard

# 定义工作流状态
class StoryWorkflowState(BaseModel):
    self.script = ""
    self.storyboard = None
    self.status = "pending"

# 创建分镜生成节点
def generate_storyboard_node(state:StoryWorkflowState) -> StoryWorkflowState:
    try:
        state.storyboard = generate_storyboard(
            script_text=state.script
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

如：上游是剧本创作智能体，下游是 文生视频+剪辑 智能。

```python

```



## 输入输出示例

输入：中文剧本

```json
{
    "script": "深夜11点，城市公寓客厅，窗外大雨滂沱。林然裹着旧羊毛毯蜷在沙发里，电视静音播放着黑白老电影。茶几上半杯凉茶已凝出水雾，旁边摊开一本旧相册。手机突然震动，屏幕亮起“未知号码”。她盯着看了三秒，指尖悬停在接听键上方，喉头轻轻滚动。终于，她按下接听，将手机贴到耳边。电话那头沉默两秒，传来一个沙哑的男声：“是我。”  林然的手指瞬间收紧，指节泛白，呼吸停滞了一瞬。  她声音微颤：“……陈默？你还好吗？”  对方停顿片刻，低声说：“我回来了。” 林然猛地坐直，瞳孔收缩，泪水在眼眶中打转。她张了张嘴，却发不出声音，只有毛毯从肩头滑落。”"
}
```

输出：结构化分镜结果

```json
{
  "fragments": [
    {
      "fragment_id": "frag_001",
      "prompt": "Cinematic wide shot of a rainy-night city apartment living room: rain-streaked window blurs vibrant neon signs outside into soft, glowing color smudges; interior lit solely by a single warm yellow floor lamp casting gentle light on a dusty vintage record player, faded movie posters on the walls, and stacked leather-bound notebooks; shallow depth of field, moody chiaroscuro lighting, film grain texture, 35mm cinematic color grading, atmospheric haze, hyper-detailed realism, slow ambient camera drift",
      "negative_prompt": "bright lighting, daylight, people, text, logos, modern furniture, clean surfaces, sharp focus everywhere, cartoonish style, low resolution, motion blur artifacts, lens flare, overexposure, cluttered composition",
      "duration": 4.0,
      "model": "runway_gen2",
      "style": "cinematic noir ambiance with nostalgic analog warmth",
      "requires_special_attention": false
    },
    {
      "fragment_id": "frag_002",
      "prompt": "Cinematic medium shot: Lin Ran curled up on a light gray fabric sofa, bare feet resting on a textured wool rug, knees covered by a faded indigo blanket with worn edges; she wears a creamy white cotton robe, hair slightly damp at the ends, her profile softly illuminated by warm floor lamp light revealing tired, serene contours; outside the window, a faint lightning flash briefly illuminates her still, delicate eyelashes — shallow depth of field, soft cinematic lighting, film grain texture, 35mm anamorphic lens aesthetic, natural skin tones, ultra-detailed fabric and textile realism, subtle ambient occlusion, moody yet intimate atmosphere.",
      "negative_prompt": "blurry, deformed hands, extra limbs, text, logos, cartoonish style, low resolution, oversaturated colors, harsh shadows, noisy grain, CGI look, anime style, smiling, motion blur, talking, open eyes blinking, daylight, cluttered background",
      "duration": 3.0,
      "model": "runway_gen2",
      "style": "Cinematic, moody, intimate, photorealistic, 35mm film aesthetic",
      "requires_special_attention": false
    }
    ......
  ],
  "global_settings": {
    "style_consistency": true,
    "use_common_negative_prompt": true
  },
  "execution_suggestions": [
    "按顺序生成片段",
    "保持相同种子值以获得一致性",
    "生成后检查片段衔接"
  ]
}
```



## 版本与展望

> 1. **依赖外部API**：LLM版本需要稳定的网络连接
> 2. **AI模型限制**：生成的视频质量受限于AI视频模型能力
> 3. **处理长剧本**：长剧本可能需要分段处理
> 4. **多语言支持**：主要针对中文优化，其他语言效果待测试

### MVP版本限制

1. **简单规则**：使用固定规则，无法处理复杂剧本结构
2. **无状态记忆能力**：只支持一次拆解，不支持超长文本的多次拆分
3. **无学习能力**：不会从用户反馈中学习优化
4. **简单切割**：视频分割简单，会有一致性、连续性、时长压缩等问题
5. **有限的自定义**：配置选项较少
6. **错误处理简单**：遇到异常可能直接失败

### 短期计划（v1.x）

1. **智能合并策略**：优化超过5秒镜头的拆分逻辑，保持一致性、连续性
2. **连续性检查**：添加角色服装、位置的基本一致性检查
3. **多模型支持**：优化Sora、Pika等模型的提示词生成
4. **规则处理**：支持基于本地规则的处理，结合 LLM + 规则两种方式
5. **英文版本**:  对英文剧本的支持

### 中期计划（v2.x）

1. **高级镜头语言**：支持更复杂的镜头类型和运动
2. **情感分析**：基于剧本内容自动调整视频风格
3. **超长剧本**：通过状态记忆，实现超长剧本解析（文档剧本）
4. **自动优化**：根据生成结果自动调整提示词
5. **批量处理**：支持多个剧本的批量处理
6. **Web界面**：提供可视化操作界面

### 长期计划（v3.x）

1. **多模态输入**：支持结合图片、音频的剧本
2. **实时预览**：生成低分辨率预览视频
3. **智能修复**：自动检测和修复连续性错误
4. **生态系统集成**：与主流视频编辑软件集成



## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目：

1. **报告问题**：在使用中遇到的问题
2. **功能建议**：希望添加的新功能
3. **代码优化**：性能优化或代码重构
4. **文档改进**：补充或修正文档