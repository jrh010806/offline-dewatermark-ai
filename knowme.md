> 2026-10-26 数据退役说明：本文保留历史记录；旧 real_samples/syn 样本与旧复跑命令不再作为当前入口。当前使用 README.md 中的固定最小回归集，旧 16 组成绩不迁移为新集合成绩。

# offline-dewatermark-ai 项目说明（C4 架构视角）

> 本文档基于对仓库源码的完整走读生成，包含 **C4 模型四级架构图**（Mermaid 渲染）与 **编译 / 运行 / 部署说明**。
> 项目本身是一个**离线智能去水印原型**：在本地工作站上对 PDF / JPG 文档做水印检测与去除，支持单张与批量处理、桌面 GUI、以及可自证的合成数据评测协议。

---

## 1. 项目概览

| 项 | 内容 |
|---|---|
| 项目名 | offline-dewatermark-ai（离线智能去水印原型） |
| 语言 / 规模 | Python（约 7.2k 行），核心包 `wm/` + 开发工具 `tools/` + 测试 `tests/` |
| 形态 | 桌面应用（PySide6）+ CLI 工具 + 评测协议，全部本地离线运行 |
| 核心思路 | 叠印物理模型 `I = s·B + t`（`s=1-α₀`，`t=α₀W`）：水印定位后用代数反混合还原底色，不透明核心（水印压字区域）交给 LaMa 深度学习修复 |
| 双引擎 | **引擎A（结构解析）**：对 PDF 中独立的透明水印对象做无损删除；**引擎B（光栅分析）**：对像素做检测 + 修复 |
| 关键约束 | 离线工作站部署；时限标书 <2 分钟/张；宁可残留不可误伤正文（保真优先） |

---

## 2. C4 架构模型

### 2.1 Level 1 — System Context（系统上下文图）

系统的外部参与者只有**操作员**（离线工作站上的使用者），系统不与任何外部服务通信（LaMa 模型首次下载除外，属部署动作而非运行时依赖）。

```mermaid
C4Context
    title 系统上下文图 - 离线智能去水印
    Person(operator, "操作员", "在离线工作站上处理带水印的 PDF/JPG 标书文档")
    System(wm, "离线智能去水印系统", "本地文档去水印：检测水印、修复底图、质量判定、批量处理")

    Rel(operator, wm, "打开文档 / 图片，查看结果，导出干净文件", "PySide6 GUI 或 CLI")
    Rel(wm, operator, "输出干净页面 PNG / PDF + manifest.json 质量报告", "本地文件系统")

    System_Ext(fs, "本地文件系统", "输入 PDF/JPG；输出结果目录 out/")
    System_Ext(fonts, "系统中文字体", "合成语料与界面渲染所需（Noto Sans CJK / 微软雅黑 / PingFang 等）")
    System_Ext(model, "LaMa ONNX 模型 (~93MB)", "可选；首次从 HuggingFace 下载后缓存本地，离线时放 models/ 目录")

    Rel(wm, fs, "读写文档与结果")
    Rel(wm, fonts, "渲染中文正文（找不到即报错，不静默回退）")
    Rel(wm, model, "onnxruntime 加载，仅引擎B修复阶段使用；缺失时显式报错")
```

### 2.2 Level 2 — Container（容器图）

单体 Python 应用，内部分四个逻辑容器：桌面 GUI、CLI/评测入口、算法管线核心、支撑工具。全部进程内运行，无网络服务。

```mermaid
C4Container
    title 容器图 - 内部逻辑划分
    Person(operator, "操作员")

    Container(app, "桌面前端 wm/app.py", "PySide6 + QThread", "页列表、批次管理、四视图(结果/原图/掩码叠加/分屏)、参数滑块、进度与指标展示；只通过 contract.py 的类型与算法层对话")
    Container(cli, "CLI 与评测 wm/cli.py + wm/e2e_test.py", "argparse", "synth(生成带真值合成集) / fixtures(生成PDF夹具) / eval(四条件协议评测) / run(处理单文件)")
    Container(core, "算法管线核心 wm/(pipeline, masks, repair, lama, structure, ocr, verdict, documents)", "numpy + OpenCV + onnxruntime + pymupdf/pypdf", "引擎A结构解析、引擎B光栅检测与修复、OCR禁区、质量判定、结果落盘")
    Container(tools, "研究/回归工具 tools/", "脚本", "blockg_probe、jpeg_floor、ocr_ab、regress_suite、verify_* 等离线验证脚本")

    ContainerDb(contract, "共享契约 wm/contract.py", "dataclass 类型", "算法轨与UI轨之间唯一的协调成本：PageImage / MaskLayer / VerdictReport / PageResult / Thresholds / JobEvent")

    Rel(operator, app, "交互")
    Rel(operator, cli, "命令行")
    Rel(app, core, "调用管线（经契约类型）")
    Rel(cli, core, "调用管线与评测")
    Rel(app, contract, "导入契约类型", "单向依赖")
    Rel(core, contract, "导入契约类型", "单向依赖")
    Rel(tools, core, "复用管线做回归验证")
```

### 2.3 Level 3 — Component（组件图）

算法管线核心内部的双引擎路由与检测-修复-判定流程：

```mermaid
C4Component
    title 组件图 - 引擎A/B 双路由与处理流程
    Container_Boundary(core, "算法管线核心") {
        Component(docs, "documents.py", "PDF/图片载入与保存", "load_pages / iter_pages(流式) / load_many / save_png；保留原PDF页号与修订标识")
        Component(struct, "structure.py", "引擎A: PDF结构解析", "probe_pdf 探测透明水印对象 → strip_pdf 无损删除；process_text_page 透明文字指令剥离")
        Component(masks, "masks.py", "引擎B: 水印定位", "单页: 直方图尖峰极点拟合/纸面锚点/墨晕保护/JPEG去块/印章与彩底通道; 跨页: estimate_crosspage 统计先验; 样式通道: paper_repeat/chromatic/repeated/periodic/document_overlay/photo_overlay")
        Component(ocr, "ocr.py", "OCR正文禁区", "RapidOCR检测框按字号/倾角/置信度筛选膨胀成保护区；可选，缺失走保守阈值")
        Component(repair, "repair.py + lama.py", "修复", "反混合还原底色 → 墨晕保护 → LaMa ONNX 深度学习重建不透明核心(512瓦片羽化拼接)")
        Component(verdict, "verdict.py", "四条件判定", "保真/残膜/识别(OCR一致性)/时限; 分解误伤文字与残留水印")
        Component(pipe, "pipeline.py", "管线调度", "双引擎路由、批量与流式逐页、结果与manifest落盘、水印图层提取")
        Component(batches, "batches.py", "批量工作线程", "BatchWorker(QThread): 批内跨页先验缓存、暂停/继续/停止、导出queue.json")
    }
    Component_Ext(contract, "contract.py", "共享契约类型（wm 包外的公共层）", "PageImage / MaskLayer / Thresholds / VerdictReport / PageResult / JobEvent；所有组件只经此通信")

    Rel(docs, pipe, "供页")
    Rel(pipe, struct, "PDF先探结构引擎")
    Rel(pipe, masks, "光栅页定位")
    Rel(pipe, ocr, "请求正文禁区", "可选")
    Rel(pipe, repair, "掩码+原图")
    Rel(pipe, verdict, "结果判定")
    Rel(batches, pipe, "批量/单张调度")
    Rel(pipe, contract, "经契约类型通信", "全体组件同")
```

**数据流（一次处理的实际顺序）：**

```
输入文件 ──► documents 载入 ──► structure 探测（PDF 且命中 → 无损删除，结束）
                                   │ 未命中
                                   ▼
                    masks 定位（单页检测；≥2页同尺寸时叠加跨页先验）
                                   │（OCR 提供正文禁区，全程保护）
                                   ▼
                    repair 修复 = alpha 反混合(纸面) → 墨晕保护 → LaMa(压字核心)
                                   │
                                   ▼
                    verdict 四条件判定 ──► write_results: 干净页PNG + 水印PNG + manifest.json
```

### 2.4 Level 4 — Code（代码层：核心类关系）

```mermaid
classDiagram
    class Thresholds {
        +float mask_alpha_min
        +float fidelity_mean_delta_max
        +float residual_coverage_max
        +float ink_iou_min
        +int time_budget_ms
        +int lama_context_margin
        +int lama_tile_overlap
        +int lama_mask_dilate
    }
    class PageImage {
        +ndarray image
        +int dpi
        +str doc_id / int page_index
        +str source_path / int source_page_index
        +tuple source_revision
        +key: str
    }
    class MaskLayer {
        +ndarray alpha
        +MaskSource source
        +float confidence
        +ndarray veil_strength / veil_offset
        +list~WatermarkInstance~ instances
        +binary(threshold) ndarray
        +coverage(threshold) float
    }
    class RegionMap {
        +ndarray label
        +mask_of(kind) ndarray
    }
    class VerdictReport {
        +str page_key / EngineKind engine
        +list~RepairName~ repairs
        +dict conditions / metrics
        +list~FailureCode~ failures
        +int elapsed_ms
        +passed: bool
        +review_needed: bool
    }
    class PageResult {
        +PageImage source / output
        +MaskLayer mask
        +VerdictReport verdict
        +ndarray extracted_watermark
        +to_dict()
    }
    class JobEvent {
        +str job_id / str page_key / JobState state
        +int index / total / elapsed_ms
    }

    PageResult --> PageImage : source/output
    PageResult --> MaskLayer : mask
    PageResult --> VerdictReport : verdict
    MaskLayer --> WatermarkInstance : instances
    JobEvent ..> PageResult : 进度事件
```

**关键枚举**（`contract.py`）：`EngineKind{STRUCTURE, RASTER}`（双引擎）、`MaskSource{STRUCTURE, CROSSPAGE, PRIOR, MODEL, HUMAN}`（掩码来源五档）、`RepairName{ALPHA_UNMIX, INPAINT, PAPER_CLEANUP, LAMA, DIFFUSION, NONE}`、`FailureCode`（8 种失败码）、`JobState`（queued/running/passed/failed/skipped/needs_review）。

---

## 3. 编译与运行说明

本项目是**纯 Python 解释执行**，无编译步骤；"编译"= 创建虚拟环境并安装依赖。

### 3.1 环境要求

| 项 | 要求 |
|---|---|
| Python | 实测 **3.14.6**（3.11/3.12 可用宽松清单降级，但须重跑 eval 对表） |
| OS | Windows（默认实测机）/ Linux / macOS；无桌面环境可 `QT_QPA_PLATFORM=offscreen` 跑测试 |
| 中文字体 | **必需**：`WM_FONTS_DIR` 环境变量，或系统目录 `C:/Windows/Fonts`、`/usr/share/fonts`、`/Library/Fonts`、`/System/Library/Fonts`；找不到直接报错 |
| 网络 | 仅首次运行 LaMa 需要（下载 ~93MB ONNX）；之后纯离线 |

> **macOS x86_64（Intel Mac）实测补充（2026-10-25）**：requirements.lock 是 Windows/Python 3.14 机的 freeze，其中两个包在 Intel Mac 上无可用 wheel：`onnxruntime==1.31.0`（官方对 macOS x86_64 只发到 1.23.2）与 `pymupdf==2.0`（要求 Python ≥3.14）。可行组合：**Python 3.13 + `requirements.txt` + `onnxruntime==1.23.2`**（其余包按 requirements.txt 精确钉死）。代码不依赖 `pymupdf4llm`/`pymupdf_layout`（lock 里的开发机残留），Python 3.14 在 Intel Mac 上不可行（onnxruntime 无任何 cp314 x86_64 版本）。安装后须重跑 `eval` 对表验收（MIGRATION.md §5）。

### 3.2 安装（四条命令）

```bash
git clone https://github.com/jrh010806/offline-dewatermark-ai.git
cd offline-dewatermark-ai
python -m venv .venv
./.venv/bin/pip install -r requirements.lock     # Windows: .venv\Scripts\pip
```

依赖分组（`requirements.txt`）：

- **经典通道（纯 CPU）**：numpy 2.5.4、opencv-python 5.0.0.93、scikit-image、scipy、pymupdf、pypdf、pillow、PySide6 6.12.0
- **LaMa 深度学习修复**：onnxruntime 1.31.0 + huggingface_hub（**不需要 torch**）
- **OCR 正文禁区**：rapidocr-onnxruntime（模型 ~8MB 随 wheel 打包，离线可用）
- **GPU 加速（可选）**：`onnxruntime-gpu` + 环境变量 `WM_LAMA_CUDA=1`

装完自检：

```bash
python -c "import numpy, cv2, skimage, fitz, PySide6, PIL; print('依赖 OK')"
python -c "from wm.lama import is_available; print('LaMa:', '可用' if is_available() else '缺失')"
python -c "from wm import ocr; print('OCR:', '可用' if ocr.available() else '未装')"
```

### 3.3 运行方式

**① 合成评测基准（带真值，可自证）**

```bash
python -m wm.cli synth --out data/syn --docs 5 --pages 6   # 生成带真值语料（约1分钟/30页）
python -m wm.cli fixtures --data data/syn --out data/pdf  # 打成光栅PDF与结构水印PDF
python -m wm.cli eval --data data/syn                      # 四条件协议评测，出成功率表
```

> 判断环境装对了的唯一标准：`eval` 数字与项目记录一致（随机种子固定 20261019 → 语料逐像素一致 → 结果可复现）。

**② 处理单个文件**

```bash
python -m wm.cli run data/pdf/doc00_tiled_text/raster_flattened.pdf --out out/run
```

**③ 不依赖中文字体的端到端自测（纯 OpenCV 渲染，随时可跑）**

```bash
python -m wm.e2e_test --out out/e2e_ort     # 5 种样式，落盘 results.json
```

**④ 桌面 GUI（多批次处理）**

```bash
python -m wm.app [文件]      # 可加 --smoke 做离屏自检
```

支持：新增/重命名/删除批次、同批同尺寸组跨页先验、逐张进度、暂停/继续/停止、按批次导出（结果 + manifest + queue.json）。参数分 `mask` 类（改后需重新定位）与 `repair` 类（可复用先验）。

**⑤ 测试与回归**

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v   # 队列与界面回归
HF_HUB_OFFLINE=1 python tools/verify_single.py                      # 单图斜向文字验证
HF_HUB_OFFLINE=1 python tools/regress_suite.py --stage after        # 全量原尺寸回归
```

### 3.4 离线部署要点

1. **LaMa 模型**：有网机器先跑一次自动下载缓存到 `~/.cache/huggingface/`；离线机拷贝到以下任一位置：
   - `~/.cache/huggingface/`
   - 项目根目录 `models/inpainting_lama_2025jan.onnx`
   - 环境变量 `WM_LAMA_ONNX=/path/to/model.onnx`
   - **缺失策略：显式抛 `LamaModelMissing`，绝不静默回退到 cv2.inpaint**
2. **OCR 模型**：随 rapidocr wheel 打包，无需下载；未装时 `text_zones()` 返回 None，检测走保守阈值，管线不中断。
3. **Linux 无桌面**：`QT_QPA_PLATFORM=offscreen` 可跑测试，但离屏平台无 CJK 字体，截图中文变方框属 Qt 正常行为。
4. **已知遗留**：`pymupdf` 为 AGPL-3.0，闭源交付需替换为 `pypdfium2`（BSD）并重跑 eval/fixtures。

### 3.5 目录与产物

| 目录 | 说明 | 在 git？ |
|---|---|---|
| `wm/` `tools/` `tests/` | 源码、工具、测试 | ✅ |
| `data/` | 合成语料、PDF 夹具（可再生，种子固定） | ❌ 生成 |
| `out/` | 处理结果（PNG + manifest.json） | ❌ 生成 |
| `models/` | LaMa ONNX 权重 | ❌ 首启下载或手工放置 |
| `docs/` | 需求分析/投标策略 | ❌ 故意不入库 |

---

## 4. 技术要点速览

- **叠印模型**：`I = s·B + t`。文档双极性给出两个可观测极点（被覆盖纸面 `s·P+t`、被覆盖墨 `t`），由 `α₀(P−W)=m` 绑定强度与颜色，只剩一个自由参数，用打分搜索确定；极点必须用直方图**尖峰**定位（抗锯齿会把分位数拖偏到文字裙边）。
- **两条保护性规则**：① 半透明叠加只有在同窗口内还存在未覆盖纸面时才"可验证"（照片渐变无水印锚点）；② 真实墨旁像素优先归文字（83–91% 假阳性落在墨笔画 3–5px 内）。
- **判定协议**（自研，需求方未给公式）：掩码质量 / 保真 / 残膜 / OCR 识别一致性 / 时限（120s/页）。无真值时 `review_needed=true`，不计入 success_rate。
- **当前状态**（README 2026-10-21）：经典通道 CPU 端到端打通；合成集掩码精确率 dot_grid 0.97 / tiled_text 0.90，整页 PSNR 33–41；LaMa 已接入，5 样式单页 13.3–29.4s；深色重字母水印与极小角标仍不可辨识。
