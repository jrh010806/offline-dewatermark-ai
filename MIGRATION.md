# 换机运行说明（MIGRATION）

目标：在任何一台新机器上，四条命令把这套代码跑到"能出评测数字"的状态。
经典通道纯 CPU 可跑；LaMa 深度学习修复可选（未装时自动回退传统算法）。

---

## 1. 哪些东西在仓库里，哪些不在

| 目录/文件 | 在 git 里？ | 换机器怎么办 |
|---|---|---|
| `wm/`、`tools/` | ✅ 在 | `git clone` 即可 |
| `requirements.txt`、`requirements.lock` | ✅ 在 | 用来装依赖 |
| `.venv/`（虚拟环境，装依赖的本地文件夹） | ❌ 不在 | **不能拷**，编译产物绑机器。新机器重建，见 §2 |
| `data/`（合成语料、PDF 夹具） | ❌ 不在 | **不用拷**，一条命令重新生成，且逐像素一致（随机种子固定为 20261019） |
| `docs/`（需求分析、路线图、进度小结） | ❌ 故意不在 | 含投标策略，**手工拷贝**这个文件夹 |
| `out/`（处理结果） | ❌ 不在 | 重跑即得，不用拷 |
| `models/`（LaMa ONNX 权重） | ❌ 不会进 git | 首次运行自动从 Hugging Face 下载（~93 MB），或手工放 `models/inpainting_lama_2025jan.onnx`。OCR 的三个模型（~8 MB）随 `rapidocr-onnxruntime` 的 wheel 一起装，**不需要下载**，见 §7 |

## 2. 新机器四条命令

```bash
git clone https://github.com/jrh010806/offline-dewatermark-ai.git
cd offline-dewatermark-ai
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.lock
```

装完先自检：

```bash
python -c "import numpy, cv2, skimage, fitz, PySide6, PIL; print('依赖 OK')"
python -c "from wm.lama import is_available; print('LaMa:', '可用' if is_available() else '缺失，修复将报错（放入 models/ 目录）')"
python -c "from wm import ocr; print('OCR:', '可用' if ocr.available() else '未装，检测走保守阈值')"
python -m wm.cli synth --out out/generated_fixtures/synthetic     # 生成带真值语料，约 30 页 / 1 分钟
python -m wm.cli fixtures                 # 打成测试用 PDF（光栅页 + 结构水印页）
python -m wm.cli eval                     # 跑四条件评测，出成功率表
```

**判断"装对了"的唯一标准是 `eval` 的数字和《进度小结》里的表一致**（同一随机种子 → 同一份语料 → 同一结果）。数字对不上就是环境有差异，别往下走。

界面：`python -m wm.app`（或双击打开一个 PDF/图片走 `python -m wm.cli run <文件>`）。

## 3. 字体（本轮已去硬编码，但仍是外部依赖）

合成语料要渲染中文正文，依赖系统中文字体。解析顺序见 `wm/synth.py:21-58`：
环境变量 `WM_FONTS_DIR` → `C:/Windows/Fonts` → `/usr/share/fonts` → `/Library/Fonts` → `wm/fonts/`（随包字体目录，目前是空的）。

- 找不到时**直接报错并提示设置 `WM_FONTS_DIR`**，不会静默回退。原因写在代码注释里：回退到 PIL 默认位图字体会把每个汉字画成方框，那样测出来的掩码指标是在测一份"没有文字的数据集"，数字全是假的。
- Linux 装一套：`apt install fonts-noto-cjk`（Noto Sans/Serif CJK，思源同源）。
- 界面字体用 `QFont.setFamilies`（字体候选列表）在微软雅黑／Noto Sans CJK／PingFang／文泉驿里按可用性挑，见 `wm/app.py:29-36`。

## 4. 平台差异

- **Windows**：默认路径，实测通过。中文路径的 PDF 输入需要再实测一轮。
- **Linux**：PySide6 需要系统库 `libEGL`／`libxkbcommon`；无桌面时用 `QT_QPA_PLATFORM=offscreen` 出图，**但离屏平台没有中日韩字体，截图里中文会变方框**——这是 Qt 的行为，不是程序的问题，要验界面外观必须在有桌面的机器上看。
- **大小写与路径分隔**：代码里全部走 `pathlib`，无手写 `\`。

## 5. Python 版本降级方案

本机实测 **Python 3.14.6** ＋ numpy 2.5.4 ＋ opencv 5.0.0.93，这套版本号很新，个别发行版仓库里没有。
若目标机只有 3.11／3.12，用宽松清单装，再回来核对 `eval` 数字：

```bash
pip install -r requirements.txt   # 只钉主版本，允许装到该 Python 上可用的构建
```

**注意**：换 numpy/opencv 小版本可能改变像素级结果（medianBlur 的实现细节、float32 求和顺序都会影响掩码）。降级后必须先跑 `eval` 对表，再决定是否接受这台机器出的数字。

## 6. LaMa 深度学习修复（WM-402）

LaMa 已接入，走 ONNX 推理（OpenCV DNN 模块），**不需要 torch**，纯 CPU 即可跑。
模型文件（~93 MB）来源：`opencv/inpainting_lama`（Hugging Face），Apache-2.0 许可。

**首次运行自动下载**：程序第一次调用 LaMa 时会从 Hugging Face 下载 ONNX 文件，
缓存到 `~/.cache/huggingface/`。之后走本地缓存，不需要网络。

**离线部署**：在有网机器上先跑一次，然后把 ONNX 文件拷到目标机的以下任一位置：
- `~/.cache/huggingface/`（默认缓存目录）
- 项目根目录下 `models/inpainting_lama_2025jan.onnx`
- 通过环境变量 `WM_LAMA_ONNX=/path/to/model.onnx` 指定

**缺失策略**：ONNX 文件找不到或加载失败时，修复阶段直接抛 `LamaModelMissing`
错误并在界面显示——**绝不静默回退**到 `cv2.inpaint`。组委会工作站断网，静默回退
会让效果倒退且无从发现，所以离线部署必须把模型文件放进项目 `models/` 目录。

**NVIDIA GPU 加速**（RTX A6000 实测机）：推理后端已是 onnxruntime，CPU 为默认。
在装有 `onnxruntime-gpu` 的机器上设环境变量 `WM_LAMA_CUDA=1` 即启用 CUDA 推理，
启动日志会打印 `providers=[...]` 供确认。

选型与许可结论（哪些模型能商用、哪些必须避开）见 `docs/AI选型与许可_2026-10-19.md`。

## 7. OCR 划禁区（检测的文字保护）

检测器分不清"淡水印"和"字的抗锯齿边"——两者都是中间灰度，所以阈值一放宽就会连字一起擦。
`wm/ocr.py` 用 OCR（光学字符识别）把"这里确实是正文文字"划成禁区，检测器在禁区外才允许放开阈值。
**注意它是反向用的**：不是识字，是划保护圈；水印本身也是字，所以只有"和正文同字号、水平、读得清"的框才算禁区，
印章那种又大又斜又淡的字不会被保护。

- 依赖：`rapidocr-onnxruntime`（RapidOCR 的 ONNX 版，复用 LaMa 已在用的 onnxruntime，不需要 paddle）。
- **模型随 wheel 打包**（检测＋方向分类＋识别三个文件共 ~8 MB），装完即离线可用，**没有首次联网下载这一步**。
- 缓存位置在 `.venv/lib/pythonX/site-packages/rapidocr_onnxruntime/models/`，随虚拟环境一起，不用手工拷。
- 成本：本机实测每页 1–6 秒（纯 CPU）。
- **可选**：没装这个包时 `text_zones()` 返回 None，检测走原来的保守阈值，管线不受影响。

自检：

```bash
python -c "from wm import ocr; print('OCR:', '可用' if ocr.available() else '未装，检测走保守阈值')"
```

## 8. 已知遗留

- `pymupdf==1.28.2` 是 **AGPL-3.0**（一种会传染的强开源许可），闭源交付需替换成 `pypdfium2`（BSD，谷歌 PDFium 的绑定）。替换后必须重跑 `eval` 与 `fixtures`，因为 PDF 渲染器换了 → 底图像素会变。
- LaMa ONNX 模型（~93 MB）首次运行需联网从 Hugging Face 下载，缓存到 `~/.cache/huggingface/`。离线部署环境需提前在有网机器上跑一次再拷缓存目录，或把文件放到 `models/` 下。
- 交付机打包（PyInstaller／Nuitka）需要在目标机上验证，本机验证不了。
