"""Watermark style identifiers and bilingual (Chinese/English) prompt parsing.

Offline by design: a fixed mapping table, no network translation API.
The same style code drives the detector routing in masks.estimate_single_page.
"""
from __future__ import annotations

# 每个样式的可用检测通道（与 masks.estimate_single_page 的级联对应）
STYLE_CHANNELS: dict[str, tuple[str, ...]] = {
    "dot_grid": ("paper_grid", "periodic"),
    "tiled_text": ("repeated", "paper_repeat"),
    "big_center": ("document_overlay", "chromatic"),
    "heavy_dark": ("dark_text", "repeated", "paper_repeat"),
    "corner_id": ("corner_id",),
}

# 中英文提示词 → 样式代码。键全部小写；按子串匹配，先长后短避免歧义。
_PROMPTS: dict[str, str] = {
    # 点阵
    "点阵": "dot_grid", "点状": "dot_grid", "圆点": "dot_grid",
    "dot grid": "dot_grid", "dot pattern": "dot_grid", "dots": "dot_grid",
    "dot_grid": "dot_grid",
    # 平铺文字
    "平铺": "tiled_text", "斜向重复": "tiled_text", "重复文字": "tiled_text",
    "tiled": "tiled_text", "repeated text": "tiled_text", "diagonal text": "tiled_text",
    "tiled_text": "tiled_text",
    # 居中大字
    "居中": "big_center", "大字": "big_center", "中央水印": "big_center",
    "big center": "big_center", "centered": "big_center", "large text": "big_center",
    "big_center": "big_center",
    # 深色重字母
    "深色": "heavy_dark", "重字母": "heavy_dark", "深色文字": "heavy_dark",
    "heavy dark": "heavy_dark", "dark text": "heavy_dark", "heavy_dark": "heavy_dark",
    # 角标
    "角标": "corner_id", "角落": "corner_id", "页脚": "corner_id",
    "corner": "corner_id", "corner id": "corner_id", "corner_id": "corner_id",
}


def parse_style_prompt(text: str | None) -> str | None:
    """把中英文提示词翻译为样式代码；无法识别时返回 None（调用方决定报错或自动路由）。

    支持直接给样式代码（tiled_text 等）或自然语言（"斜向平铺文字" /
    "tiled text watermark"）。大小写与首尾空白不敏感。
    """
    if not text:
        return None
    q = text.strip().lower()
    if not q:
        return None
    if q in STYLE_CHANNELS:
        return q
    for key in sorted(_PROMPTS, key=len, reverse=True):
        if key in q:
            return _PROMPTS[key]
    return None


def translate_prompt(text: str) -> dict[str, str | None]:
    """供 UI/文档使用的中英文互译结果：返回样式代码与英文、中文名称。"""
    code = parse_style_prompt(text)
    names = {
        "dot_grid": ("dot grid watermark", "点阵水印"),
        "tiled_text": ("tiled repeated text watermark", "斜向平铺重复文字水印"),
        "big_center": ("large centered text watermark", "居中大号文字水印"),
        "heavy_dark": ("heavy dark lettering watermark", "深色重字母水印"),
        "corner_id": ("corner identifier watermark", "角标水印"),
    }
    en, zh = names.get(code, (None, None)) if code else (None, None)
    return {"input": text, "style": code, "english": en, "chinese": zh}
