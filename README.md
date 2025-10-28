# 剧本分镜智能体 (Script-to-Shot AI Agent)

一个基于多智能体协作的AI系统，能够将剧本智能拆分为短视频脚本单元，生成高质量分镜描述，并保证叙事连续性。支持多种AI提供商，具有强大的可扩展性和易用性。

> 将**一段自然语言中文剧本** → 自动拆分为 **N 个 5 秒分镜**，每个分镜包含：
>
> - 中文画面描述（供人读）
> - 英文 AI 视频提示词（供 Runway/Pika/Sora 使用）
> - 角色连续性锚点（防漂移）
> - 镜头语言建议



## 核心功能

<img src="data:image/svg+xml;utf8,<svg id%3D%22mermaid-908164f6-4902-4670-861f-d89f7c52a5df%22 width%3D%22100%%22 xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22 style%3D%22max-width%3A 1431.697509765625px%3B%22 viewBox%3D%22-8.000007629394531 -8 1431.697509765625 240.6344451904297%22 role%3D%22graphics-document document%22 aria-roledescription%3D%22flowchart-v2%22><style>%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df{font-family%3A%22trebuchet ms%22%2Cverdana%2Carial%2Csans-serif%3Bfont-size%3A16px%3Bfill%3A%23333%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .error-icon{fill%3A%23552222%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .error-text{fill%3A%23552222%3Bstroke%3A%23552222%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .edge-thickness-normal{stroke-width%3A2px%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .edge-thickness-thick{stroke-width%3A3.5px%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .edge-pattern-solid{stroke-dasharray%3A0%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .edge-pattern-dashed{stroke-dasharray%3A3%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .edge-pattern-dotted{stroke-dasharray%3A2%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .marker{fill%3A%23333333%3Bstroke%3A%23333333%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .marker.cross{stroke%3A%23333333%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df svg{font-family%3A%22trebuchet ms%22%2Cverdana%2Carial%2Csans-serif%3Bfont-size%3A16px%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .label{font-family%3A%22trebuchet ms%22%2Cverdana%2Carial%2Csans-serif%3Bcolor%3A%23333%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .cluster-label text{fill%3A%23333%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .cluster-label span%2C%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df p{color%3A%23333%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .label text%2C%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df span%2C%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df p{fill%3A%23333%3Bcolor%3A%23333%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .node rect%2C%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .node circle%2C%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .node ellipse%2C%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .node polygon%2C%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .node path{fill%3A%23ECECFF%3Bstroke%3A%239370DB%3Bstroke-width%3A1px%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .flowchart-label text{text-anchor%3Amiddle%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .node .katex path{fill%3A%23000%3Bstroke%3A%23000%3Bstroke-width%3A1px%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .node .label{text-align%3Acenter%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .node.clickable{cursor%3Apointer%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .arrowheadPath{fill%3A%23333333%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .edgePath .path{stroke%3A%23333333%3Bstroke-width%3A2.0px%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .flowchart-link{stroke%3A%23333333%3Bfill%3Anone%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .edgeLabel{background-color%3A%23e8e8e8%3Btext-align%3Acenter%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .edgeLabel rect{opacity%3A0.5%3Bbackground-color%3A%23e8e8e8%3Bfill%3A%23e8e8e8%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .labelBkg{background-color%3Argba(232%2C 232%2C 232%2C 0.5)%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .cluster rect{fill%3A%23ffffde%3Bstroke%3A%23aaaa33%3Bstroke-width%3A1px%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .cluster text{fill%3A%23333%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .cluster span%2C%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df p{color%3A%23333%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df div.mermaidTooltip{position%3Aabsolute%3Btext-align%3Acenter%3Bmax-width%3A200px%3Bpadding%3A2px%3Bfont-family%3A%22trebuchet ms%22%2Cverdana%2Carial%2Csans-serif%3Bfont-size%3A12px%3Bbackground%3Ahsl(80%2C 100%%2C 96.2745098039%)%3Bborder%3A1px solid %23aaaa33%3Bborder-radius%3A2px%3Bpointer-events%3Anone%3Bz-index%3A100%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df .flowchartTitleText{text-anchor%3Amiddle%3Bfont-size%3A18px%3Bfill%3A%23333%3B}%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df %3Aroot{--mermaid-font-family%3A%22trebuchet ms%22%2Cverdana%2Carial%2Csans-serif%3B}<%2Fstyle><g><marker id%3D%22mermaid-908164f6-4902-4670-861f-d89f7c52a5df_flowchart-pointEnd%22 class%3D%22marker flowchart%22 viewBox%3D%220 0 10 10%22 refX%3D%226%22 refY%3D%225%22 markerUnits%3D%22userSpaceOnUse%22 markerWidth%3D%2212%22 markerHeight%3D%2212%22 orient%3D%22auto%22><path d%3D%22M 0 0 L 10 5 L 0 10 z%22 class%3D%22arrowMarkerPath%22 style%3D%22stroke-width%3A 1%3B stroke-dasharray%3A 1%2C 0%3B%22><%2Fpath><%2Fmarker><marker id%3D%22mermaid-908164f6-4902-4670-861f-d89f7c52a5df_flowchart-pointStart%22 class%3D%22marker flowchart%22 viewBox%3D%220 0 10 10%22 refX%3D%224.5%22 refY%3D%225%22 markerUnits%3D%22userSpaceOnUse%22 markerWidth%3D%2212%22 markerHeight%3D%2212%22 orient%3D%22auto%22><path d%3D%22M 0 5 L 10 10 L 10 0 z%22 class%3D%22arrowMarkerPath%22 style%3D%22stroke-width%3A 1%3B stroke-dasharray%3A 1%2C 0%3B%22><%2Fpath><%2Fmarker><marker id%3D%22mermaid-908164f6-4902-4670-861f-d89f7c52a5df_flowchart-circleEnd%22 class%3D%22marker flowchart%22 viewBox%3D%220 0 10 10%22 refX%3D%2211%22 refY%3D%225%22 markerUnits%3D%22userSpaceOnUse%22 markerWidth%3D%2211%22 markerHeight%3D%2211%22 orient%3D%22auto%22><circle cx%3D%225%22 cy%3D%225%22 r%3D%225%22 class%3D%22arrowMarkerPath%22 style%3D%22stroke-width%3A 1%3B stroke-dasharray%3A 1%2C 0%3B%22><%2Fcircle><%2Fmarker><marker id%3D%22mermaid-908164f6-4902-4670-861f-d89f7c52a5df_flowchart-circleStart%22 class%3D%22marker flowchart%22 viewBox%3D%220 0 10 10%22 refX%3D%22-1%22 refY%3D%225%22 markerUnits%3D%22userSpaceOnUse%22 markerWidth%3D%2211%22 markerHeight%3D%2211%22 orient%3D%22auto%22><circle cx%3D%225%22 cy%3D%225%22 r%3D%225%22 class%3D%22arrowMarkerPath%22 style%3D%22stroke-width%3A 1%3B stroke-dasharray%3A 1%2C 0%3B%22><%2Fcircle><%2Fmarker><marker id%3D%22mermaid-908164f6-4902-4670-861f-d89f7c52a5df_flowchart-crossEnd%22 class%3D%22marker cross flowchart%22 viewBox%3D%220 0 11 11%22 refX%3D%2212%22 refY%3D%225.2%22 markerUnits%3D%22userSpaceOnUse%22 markerWidth%3D%2211%22 markerHeight%3D%2211%22 orient%3D%22auto%22><path d%3D%22M 1%2C1 l 9%2C9 M 10%2C1 l -9%2C9%22 class%3D%22arrowMarkerPath%22 style%3D%22stroke-width%3A 2%3B stroke-dasharray%3A 1%2C 0%3B%22><%2Fpath><%2Fmarker><marker id%3D%22mermaid-908164f6-4902-4670-861f-d89f7c52a5df_flowchart-crossStart%22 class%3D%22marker cross flowchart%22 viewBox%3D%220 0 11 11%22 refX%3D%22-1%22 refY%3D%225.2%22 markerUnits%3D%22userSpaceOnUse%22 markerWidth%3D%2211%22 markerHeight%3D%2211%22 orient%3D%22auto%22><path d%3D%22M 1%2C1 l 9%2C9 M 10%2C1 l -9%2C9%22 class%3D%22arrowMarkerPath%22 style%3D%22stroke-width%3A 2%3B stroke-dasharray%3A 1%2C 0%3B%22><%2Fpath><%2Fmarker><g class%3D%22root%22><g class%3D%22clusters%22><%2Fg><g class%3D%22edgePaths%22><path d%3D%22M190.979%2C112.317L195.146%2C112.317C199.312%2C112.317%2C207.646%2C112.317%2C215.096%2C112.317C222.546%2C112.317%2C229.112%2C112.317%2C232.396%2C112.317L235.679%2C112.317%22 id%3D%22L-A-B-0%22 class%3D%22 edge-thickness-normal edge-pattern-solid flowchart-link LS-A LE-B%22 style%3D%22fill%3Anone%3B%22 marker-end%3D%22url(%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df_flowchart-pointEnd)%22><%2Fpath><path d%3D%22M345.412%2C112.317L349.578%2C112.317C353.745%2C112.317%2C362.078%2C112.317%2C369.612%2C112.383C377.145%2C112.449%2C383.879%2C112.581%2C387.246%2C112.647L390.613%2C112.713%22 id%3D%22L-B-C-0%22 class%3D%22 edge-thickness-normal edge-pattern-solid flowchart-link LS-B LE-C%22 style%3D%22fill%3Anone%3B%22 marker-end%3D%22url(%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df_flowchart-pointEnd)%22><%2Fpath><path d%3D%22M620.546%2C112.817L624.63%2C112.734C628.713%2C112.651%2C636.88%2C112.484%2C644.246%2C112.401C651.613%2C112.317%2C658.18%2C112.317%2C661.463%2C112.317L664.746%2C112.317%22 id%3D%22L-C-D-0%22 class%3D%22 edge-thickness-normal edge-pattern-solid flowchart-link LS-C LE-D%22 style%3D%22fill%3Anone%3B%22 marker-end%3D%22url(%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df_flowchart-pointEnd)%22><%2Fpath><path d%3D%22M873.958%2C112.317L878.125%2C112.317C882.291%2C112.317%2C890.625%2C112.317%2C898.075%2C112.317C905.525%2C112.317%2C912.091%2C112.317%2C915.375%2C112.317L918.658%2C112.317%22 id%3D%22L-D-E-0%22 class%3D%22 edge-thickness-normal edge-pattern-solid flowchart-link LS-D LE-E%22 style%3D%22fill%3Anone%3B%22 marker-end%3D%22url(%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df_flowchart-pointEnd)%22><%2Fpath><path d%3D%22M1086.947%2C92.824L1092.144%2C91.531C1097.342%2C90.239%2C1107.736%2C87.655%2C1116.216%2C86.363C1124.697%2C85.07%2C1131.264%2C85.07%2C1134.547%2C85.07L1137.83%2C85.07%22 id%3D%22L-E-F-0%22 class%3D%22 edge-thickness-normal edge-pattern-solid flowchart-link LS-E LE-F%22 style%3D%22fill%3Anone%3B%22 marker-end%3D%22url(%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df_flowchart-pointEnd)%22><%2Fpath><path d%3D%22M1222.706%2C74.239L1229.54%2C72.378C1236.373%2C70.518%2C1250.041%2C66.797%2C1262.825%2C64.937C1275.609%2C63.077%2C1287.509%2C63.077%2C1293.46%2C63.077L1299.41%2C63.077%22 id%3D%22L-F-G-0%22 class%3D%22 edge-thickness-normal edge-pattern-solid flowchart-link LS-F LE-G%22 style%3D%22fill%3Anone%3B%22 marker-end%3D%22url(%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df_flowchart-pointEnd)%22><%2Fpath><path d%3D%22M1218.314%2C104.564L1225.88%2C108.731C1233.445%2C112.897%2C1248.577%2C121.231%2C1264.792%2C127.84C1281.007%2C134.449%2C1298.306%2C139.333%2C1306.956%2C141.776L1315.605%2C144.218%22 id%3D%22L-F-H-0%22 class%3D%22 edge-thickness-normal edge-pattern-solid flowchart-link LS-F LE-H%22 style%3D%22fill%3Anone%3B%22 marker-end%3D%22url(%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df_flowchart-pointEnd)%22><%2Fpath><path d%3D%22M1320.706%2C163.359L1311.206%2C164.934C1301.707%2C166.508%2C1282.707%2C169.658%2C1259.743%2C171.233C1236.778%2C172.808%2C1209.848%2C172.808%2C1185.585%2C172.808C1161.322%2C172.808%2C1139.726%2C172.808%2C1117.323%2C166.402C1094.92%2C159.996%2C1071.71%2C147.184%2C1060.105%2C140.778L1048.499%2C134.372%22 id%3D%22L-H-E-0%22 class%3D%22 edge-thickness-normal edge-pattern-solid flowchart-link LS-H LE-E%22 style%3D%22fill%3Anone%3B%22 marker-end%3D%22url(%23mermaid-908164f6-4902-4670-861f-d89f7c52a5df_flowchart-pointEnd)%22><%2Fpath><%2Fg><g class%3D%22edgeLabels%22><g class%3D%22edgeLabel%22><g class%3D%22label%22 transform%3D%22translate(0%2C 0)%22><foreignObject width%3D%220%22 height%3D%220%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22edgeLabel%22><%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><g class%3D%22edgeLabel%22><g class%3D%22label%22 transform%3D%22translate(0%2C 0)%22><foreignObject width%3D%220%22 height%3D%220%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22edgeLabel%22><%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><g class%3D%22edgeLabel%22><g class%3D%22label%22 transform%3D%22translate(0%2C 0)%22><foreignObject width%3D%220%22 height%3D%220%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22edgeLabel%22><%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><g class%3D%22edgeLabel%22><g class%3D%22label%22 transform%3D%22translate(0%2C 0)%22><foreignObject width%3D%220%22 height%3D%220%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22edgeLabel%22><%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><g class%3D%22edgeLabel%22><g class%3D%22label%22 transform%3D%22translate(0%2C 0)%22><foreignObject width%3D%220%22 height%3D%220%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22edgeLabel%22><%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><g class%3D%22edgeLabel%22 transform%3D%22translate(1263.7079334259033%2C 63.076677322387695)%22><g class%3D%22label%22 transform%3D%22translate(-16.002099990844727%2C -11.993697166442871)%22><foreignObject width%3D%2232.00419998168945%22 height%3D%2223.987394332885742%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22edgeLabel%22>通过<%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><g class%3D%22edgeLabel%22 transform%3D%22translate(1263.7079334259033%2C 129.56407070159912)%22><g class%3D%22label%22 transform%3D%22translate(-16.002099990844727%2C -11.993697166442871)%22><foreignObject width%3D%2232.00419998168945%22 height%3D%2223.987394332885742%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22edgeLabel%22>失败<%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><g class%3D%22edgeLabel%22><g class%3D%22label%22 transform%3D%22translate(0%2C 0)%22><foreignObject width%3D%220%22 height%3D%220%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22edgeLabel%22><%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><%2Fg><g class%3D%22nodes%22><g class%3D%22node default default flowchart-label%22 id%3D%22flowchart-A-16%22 data-node%3D%22true%22 data-id%3D%22A%22 transform%3D%22translate(95.48949432373047%2C 112.31722259521484)%22><rect class%3D%22basic label-container%22 style%3D%22%22 rx%3D%220%22 ry%3D%220%22 x%3D%22-95.48949432373047%22 y%3D%22-19.49369716644287%22 width%3D%22190.97898864746094%22 height%3D%2238.98739433288574%22><%2Frect><g class%3D%22label%22 style%3D%22%22 transform%3D%22translate(-87.98949432373047%2C -11.993697166442871)%22><rect><%2Frect><foreignObject width%3D%22175.97898864746094%22 height%3D%2223.987394332885742%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22nodeLabel%22>用户输入：整段中文剧本<%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><g class%3D%22node default default flowchart-label%22 id%3D%22flowchart-B-17%22 data-node%3D%22true%22 data-id%3D%22B%22 transform%3D%22translate(293.19537353515625%2C 112.31722259521484)%22><rect class%3D%22basic label-container%22 style%3D%22%22 rx%3D%225%22 ry%3D%225%22 x%3D%22-52.21638488769531%22 y%3D%22-19.49369716644287%22 width%3D%22104.43276977539062%22 height%3D%2238.98739433288574%22><%2Frect><g class%3D%22label%22 style%3D%22%22 transform%3D%22translate(-44.71638488769531%2C -11.993697166442871)%22><rect><%2Frect><foreignObject width%3D%2289.43276977539062%22 height%3D%2223.987394332885742%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22nodeLabel%22>Parser Agent<%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><g class%3D%22node default default flowchart-label%22 id%3D%22flowchart-C-19%22 data-node%3D%22true%22 data-id%3D%22C%22 transform%3D%22translate(507.7289810180664%2C 112.31722259521484)%22><polygon points%3D%22112.31722164154053%2C0 224.63444328308105%2C-112.31722164154053 112.31722164154053%2C-224.63444328308105 0%2C-112.31722164154053%22 class%3D%22label-container%22 transform%3D%22translate(-112.31722164154053%2C112.31722164154053)%22 style%3D%22%22><%2Fpolygon><g class%3D%22label%22 style%3D%22%22 transform%3D%22translate(-85.32352447509766%2C -11.993697166442871)%22><rect><%2Frect><foreignObject width%3D%22170.6470489501953%22 height%3D%2223.987394332885742%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22nodeLabel%22>Temporal Planner Agent<%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><g class%3D%22node default default flowchart-label%22 id%3D%22flowchart-D-21%22 data-node%3D%22true%22 data-id%3D%22D%22 transform%3D%22translate(772.002082824707%2C 112.31722259521484)%22><rect class%3D%22basic label-container%22 style%3D%22%22 rx%3D%220%22 ry%3D%220%22 x%3D%22-101.95587921142578%22 y%3D%22-19.49369716644287%22 width%3D%22203.91175842285156%22 height%3D%2238.98739433288574%22><%2Frect><g class%3D%22label%22 style%3D%22%22 transform%3D%22translate(-94.45587921142578%2C -11.993697166442871)%22><rect><%2Frect><foreignObject width%3D%22188.91175842285156%22 height%3D%2223.987394332885742%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22nodeLabel%22>Continuity Guardian Agent<%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><g class%3D%22node default default flowchart-label%22 id%3D%22flowchart-E-23%22 data-node%3D%22true%22 data-id%3D%22E%22 transform%3D%22translate(1008.5440826416016%2C 112.31722259521484)%22><rect class%3D%22basic label-container%22 style%3D%22%22 rx%3D%220%22 ry%3D%220%22 x%3D%22-84.58612060546875%22 y%3D%22-19.49369716644287%22 width%3D%22169.1722412109375%22 height%3D%2238.98739433288574%22><%2Frect><g class%3D%22label%22 style%3D%22%22 transform%3D%22translate(-77.08612060546875%2C -11.993697166442871)%22><rect><%2Frect><foreignObject width%3D%22154.1722412109375%22 height%3D%2223.987394332885742%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22nodeLabel%22>Shot Generator Agent<%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><g class%3D%22node default default flowchart-label%22 id%3D%22flowchart-F-25%22 data-node%3D%22true%22 data-id%3D%22F%22 transform%3D%22translate(1182.9180183410645%2C 85.07037448883057)%22><rect class%3D%22basic label-container%22 style%3D%22%22 rx%3D%220%22 ry%3D%220%22 x%3D%22-39.78781509399414%22 y%3D%22-19.49369716644287%22 width%3D%2279.57563018798828%22 height%3D%2238.98739433288574%22><%2Frect><g class%3D%22label%22 style%3D%22%22 transform%3D%22translate(-32.28781509399414%2C -11.993697166442871)%22><rect><%2Frect><foreignObject width%3D%2264.57563018798828%22 height%3D%2223.987394332885742%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22nodeLabel%22>QA Agent<%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><g class%3D%22node default default flowchart-label%22 id%3D%22flowchart-G-27%22 data-node%3D%22true%22 data-id%3D%22G%22 transform%3D%22translate(1360.2037239074707%2C 63.076677322387695)%22><rect class%3D%22basic label-container%22 style%3D%22%22 rx%3D%220%22 ry%3D%220%22 x%3D%22-55.493690490722656%22 y%3D%22-19.49369716644287%22 width%3D%22110.98738098144531%22 height%3D%2238.98739433288574%22><%2Frect><g class%3D%22label%22 style%3D%22%22 transform%3D%22translate(-47.993690490722656%2C -11.993697166442871)%22><rect><%2Frect><foreignObject width%3D%2295.98738098144531%22 height%3D%2223.987394332885742%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22nodeLabel%22>输出分镜序列<%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><g class%3D%22node default default flowchart-label%22 id%3D%22flowchart-H-29%22 data-node%3D%22true%22 data-id%3D%22H%22 transform%3D%22translate(1360.2037239074707%2C 156.8109188079834)%22><rect class%3D%22basic label-container%22 style%3D%22%22 rx%3D%220%22 ry%3D%220%22 x%3D%22-39.49789810180664%22 y%3D%22-19.49369716644287%22 width%3D%2278.99579620361328%22 height%3D%2238.98739433288574%22><%2Frect><g class%3D%22label%22 style%3D%22%22 transform%3D%22translate(-31.99789810180664%2C -11.993697166442871)%22><rect><%2Frect><foreignObject width%3D%2263.99579620361328%22 height%3D%2223.987394332885742%22><div xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22 style%3D%22display%3A inline-block%3B white-space%3A nowrap%3B%22><span class%3D%22nodeLabel%22>局部重试<%2Fspan><%2Fdiv><%2FforeignObject><%2Fg><%2Fg><%2Fg><%2Fg><%2Fg><%2Fsvg>" alt="SVG content" />

- **智能剧本解析**：自动识别场景、对话和动作指令，支持自然语言和JSON格式
- **精准时序规划**：按短视频粒度智能切分内容，优化叙事节奏
- **连续性守护**：确保相邻分镜间角色状态、场景和情节的一致性
- **高质量分镜生成**：生成详细的中文画面描述和英文AI提示词，包含镜头角度、角色状态等
- **多模型支持**：兼容OpenAI、Qwen、DeepSeek、Ollama等多种AI提供商
- **自动重试机制**：请求失败时自动重试，提高系统稳定性
- **质量审查**：自动检查分镜质量和连续性问题，提供优化建议

## 技术架构

项目采用多智能体协作架构，基于以下技术栈：

- **Python 3.10+**：核心开发语言
- **FastAPI**：高性能Web框架
- **LangChain + LangGraph**：工作流编排和智能体管理
- **多模型支持**：兼容OpenAI、Qwen、DeepSeek、Ollama等
- **Pydantic**：数据验证和设置管理
- **环境变量配置**：灵活的配置管理

## 快速上手

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/HengLine/tool-storyboard-agent.git
cd tool-storyboard-agent

# 直接运行，自动创建虚拟环境
python .\start_app.py

# 或者手动创建虚拟环境
    python -m venv .venv
    # 激活虚拟环境 (Windows)
    .venv\Scripts\activate
    # 激活虚拟环境 (Linux/Mac)
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
QWEN_API_KEY=your_qwen_api_key
QWEN_BASE_URL=https://api.example.com/v1

# 或 OPENAI_API_KEY=your_openai_api_key
# 或 DEEPSEEK_API_KEY=your_deepseek_api_key

# 可选：设置超时时间和重试次数
AI_API_TIMEOUT=60
AI_RETRY_COUNT=3
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
场景：咖啡馆内
小明坐在窗边，看着窗外的雨。
小红：你看起来心情不太好。
小明：嗯，工作上遇到了一些问题。
小红：别担心，一切都会好起来的。
"""

# 生成分镜
result = generate_storyboard(script_text)
print(f"生成了 {result['total_shots']} 个分镜")
for shot in result['shots']:
    print(f"\n分镜 {shot['shot_id']}:")
    print(f"时间: {shot['start_time']}-{shot['end_time']}s")
    print(f"描述: {shot['description']}")
```

### 2. API接口调用

启动服务后，可以通过HTTP接口调用：

```bash
curl -X POST http://localhost:8000/api/generate_storyboard \
  -H "Content-Type: application/json" \
  -d '{"script_text": "场景：咖啡馆内\n小明坐在窗边...", "style": "realistic"}'
```

### 3. 集成到其他系统

#### 集成到Web应用

```python
# Flask示例
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

if __name__ == '__main__':
    app.run(debug=True)
```

#### 集成到LangChain工作流

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from hengline.client.client_factory import get_ai_client

# 获取配置的AI客户端
llm = get_ai_client()

# 创建LangChain链
prompt = PromptTemplate(
    input_variables=["story"],
    template="总结这个故事：{story}"
)
chain = LLMChain(llm=llm, prompt=prompt)

# 使用链
result = chain.run(story="小明和小红在咖啡馆的对话...")
print(result)
```

#### 集成到A2A系统

```python
# A2A Agent示例
from a2a import Agent, Message
from hengline.generate_agent import generate_storyboard

class StoryboardAgent(Agent):
    def process_message(self, message: Message) -> Message:
        # 处理传入的剧本消息
        script = message.content
        storyboard = generate_storyboard(script)
        
        # 返回分镜结果
        return Message(
            content=storyboard,
            type="storyboard_result"
        )

# 注册和使用Agent
agent = StoryboardAgent(name="storyboard_agent")
```

## 输入输出

```python
# 自定义风格和时长
result = generate_storyboard(
    script_text,
    style="cinematic",  # 可选: realistic, anime, cinematic, cartoon
    duration_per_shot=8,  # 每段目标时长（秒）
    prev_continuity_state=None  # 用于长剧本续生成
)
```

生成的分镜结果为结构化JSON，包含以下核心字段：

```json
{
  "total_shots": 3,              // 生成的分镜总数
  "storyboard_title": "咖啡馆对话", // 分镜标题
  "shots": [
    {
      "shot_id": "shot_001",    // 分镜ID
      "start_time": 0.0,         // 开始时间（秒）
      "end_time": 5.0,           // 结束时间（秒）
      "duration": 5.0,           // 分镜时长
      "description": "小明坐在咖啡馆窗边...", // 中文画面描述
      "prompt_en": "A man sitting by the window...", // 英文AI提示词
      "characters": ["小明"],    // 角色列表
      "dialogue": "",           // 对话内容
      "camera_angle": "medium shot", // 镜头角度
      "continuity_anchors": ["小明位置:窗边", "天气:下雨"] // 连续性锚点
    },
    // 更多分镜...
  ],
  "status": "success",          // 生成状态
  "warnings": []                 // 警告信息
}
```



## 配置说明

系统配置支持两种方式：配置文件和环境变量（优先级更高）。

### 环境变量配置

关键环境变量：

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| AI_PROVIDER | AI提供商名称（openai/qwen/deepseek/ollama） | openai |
| OPENAI_API_KEY | OpenAI API密钥 | - |
| QWEN_API_KEY | 文心一言API密钥 | - |
| DEEPSEEK_API_KEY | DeepSeek API密钥 | - |
| AI_API_TIMEOUT | API请求超时时间（秒） | 60 |
| AI_RETRY_COUNT | 请求失败重试次数 | 3 |
| AI_TEMPERATURE | 生成温度参数 | 0.7 |
| AI_MAX_TOKENS | 最大生成令牌数 | 2000 |

### 配置文件

`config/config.json` 包含默认配置，可通过环境变量覆盖。

## 实际应用场景

### 短视频内容创作
- 将小说章节转换为短视频分镜脚本
- 为广告创意生成详细的镜头规划
- 自动将剧本拆分为社交媒体短视频格式

### 影视前期制作辅助
- 快速生成剧本的视觉化预览
- 辅助导演进行镜头规划和调度
- 为分镜头绘制提供详细参考

### 教育培训应用
- 为教学内容创建情景化视频脚本
- 将复杂概念通过分镜形式直观呈现
- 辅助培训视频的标准化制作

## 最佳实践

1. **剧本格式优化**
   - 使用明确的场景标识和角色对白格式
   - 避免过于冗长的描述，保持每个场景的焦点
   - 为重要动作和情感变化添加明确标记

2. **参数调优**
   - 对于对话密集型内容，可适当延长`duration_per_shot`
   - 情感细腻的场景推荐使用`cinematic`风格
   - 动作场景可选择`realistic`风格获得更准确的描述

3. **性能优化**
   - 对于长剧本，建议分段处理并使用`prev_continuity_state`保持连贯性
   - 根据服务器资源调整`AI_RETRY_COUNT`参数
   - 生产环境中推荐使用`gpt-4o`或同等性能模型

## 故障排除

### 常见问题及解决方案

1. **API密钥错误**
   - 检查环境变量中的API密钥是否正确设置
   - 确保密钥未过期，并有足够的使用额度
   - 验证AI_PROVIDER与密钥类型是否匹配

2. **分镜生成失败**
   - 检查剧本格式是否规范，尝试简化复杂描述
   - 增加`AI_RETRY_COUNT`参数值
   - 查看日志文件获取详细错误信息

3. **连续性问题**
   - 确保相邻场景描述包含足够的上下文信息
   - 对于长剧本，使用分段处理并传递连续性状态
   - 检查`continuity_anchors`字段是否正确捕获关键信息

4. **性能问题**
   - 降低模型温度参数可提高响应速度
   - 减少单次处理的剧本长度
   - 优化系统资源分配

## 许可证

MIT License

## 贡献指南

欢迎提交Issue和Pull Request！贡献前请确保：

1. 遵循现有代码风格和架构
2. 为新功能添加适当的测试用例
3. 更新相关文档

## 联系方式

如有问题或建议，请提交GitHub Issue或联系项目维护团队。