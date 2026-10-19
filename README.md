# 离线智能去水印（需求 7009111111）原型

离线工作站本地部署的可视化文档去水印原型：PDF / JPG 单张与批量处理，复杂样式（字母、数字、点状图案）。

## 快速开始

```bash
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt

# 生成带真值的合成样本集（手上没有真实样本，这是唯一可自证的基准）
./.venv/Scripts/python.exe -m wm.cli synth --out data/syn --docs 5 --pages 6

# 把合成集打成两种 PDF 形态：压平光栅页（引擎B）与独立透明水印图层（引擎A）
./.venv/Scripts/python.exe -m wm.cli fixtures --data data/syn --out data/pdf

# 四条件协议评测（掩码质量 / 保真 / 残膜 / OCR 一致性 / 时限）
./.venv/Scripts/python.exe -m wm.cli eval --data data/syn

# 处理一个文件或图片
./.venv/Scripts/python.exe -m wm.cli run data/pdf/doc00_tiled_text/raster_flattened.pdf --out out/run

# 打开桌面前端
./.venv/Scripts/python.exe -m wm.app data/pdf/doc00_tiled_text/raster_flattened.pdf
```

## 代码地图

| 文件 | 职责 |
|---|---|
| `wm/contract.py` | 接口契约：算法轨与 UI 轨唯一的协调成本 |
| `wm/synth.py` | 带真值合成集（含印章、点线表格、照片渐变等**不可删除内容**陷阱） |
| `wm/masks.py` | 水印定位：直方图尖峰极点、纸面锚点门、墨晕保护、跨页统计先验 |
| `wm/repair.py` | 修复：代数反混合 `B=(I-βt)/(1-β(1-s))`，可选修补 |
| `wm/verdict.py` | 判定：保真 / 残膜 / 识别 / 时限四条件，并分解"误伤文字"与"残留水印" |
| `wm/structure.py` | 引擎A：探测并删除 PDF 中的独立水印对象（无损） |
| `wm/pipeline.py` | 双引擎路由与逐页处理、结果落盘、单层水印提取 |
| `wm/app.py` | PySide6 前端：页列表、结果/原图/掩码叠加/分屏四视图、指标与进度 |
| `wm/cli.py` | `synth` / `eval` / `fixtures` / `run` |

## 技术要点

叠印物理模型 `I = s·B + t`（`s = 1-α₀`，`t = α₀W`）。文档双极性给出两个**可观测**极点：被覆盖纸面 `= s·P + t`、被覆盖墨 `= t`；由 `α₀(P−W) = m` 把强度与颜色绑定，只剩一个自由参数，因此可用打分搜索确定。极点必须用直方图**尖峰**定位——抗锯齿把偏差填成连续谱，分位数会落在文字裙边上。

两条保护性规则来自实测：半透明叠加只有当同一窗口内还存在未覆盖纸面时才"可验证"（照片渐变色块与水印一样逐页不变，但没有纸面锚点）；真实墨旁的像素优先归给文字而不是水印（83–91% 的假阳性落在墨笔画 3–5 px 内）。

## 当前状态

- 经典通道（纯 CPU）已打通端到端；掩码精确率 dot_grid 0.97 / tiled_text 0.90，整页 PSNR 33–41。
- 唯一硬缺口：水印**压在文字或图案上**的那部分像素无法用代数反混合还原，需要学习式重建（路线图 WM-402）。
- 深色重字母水印与极小角标在单页色调通道不可辨识，需走结构解析或样式库通道。
- 指标口径是本项目自定义（招标原文未给成功率公式），需与需求方书面确认。

详细功能矩阵（40 条编号，M1–M8）、非功能需求、UI 页面清单与风险登记见 `docs/交付物路线图.md`。
