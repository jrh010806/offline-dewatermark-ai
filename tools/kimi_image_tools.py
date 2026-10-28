"""Project-scoped image tools for Kimi; image edits use the bundled imagegen CLI."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime

from PIL import Image, ImageChops, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / ".local/share/kimi-image-tools/venv/bin/python"
SECRETS = Path.home() / ".config/kimi-image-tools/openai_api_key"
IMAGEGEN = Path(os.environ.get(
    "KIMI_IMAGEGEN_CLI",
    str(Path.home() / ".codex/skills/.system/imagegen/scripts/image_gen.py"),
))
MODEL = "gpt-image-2.5-sunburst"
PROMPT = """Use case: precise-object-edit.
Image 1 is the original image to restore. Transparent pixels in the mask mark
the watermark strokes to remove. Use the complete image to understand objects,
geometry, perspective, lighting and texture. Reconstruct the content beneath
the watermark coherently. Preserve all visible original details, composition,
legitimate text, numbers, table lines and object positions. Do not invent new
objects, text or numbers. Keep the image framing and pixel alignment unchanged.
The padded border is context only, not part of the original image.
"""


def project_file(value: str) -> Path:
    path = Path(value).expanduser()
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("请先把输入图片复制到当前项目目录，再调用此工具。")
    if not path.is_file():
        raise ValueError(f"找不到输入文件：{path}")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        raise ValueError("只接收图片；PDF 请先按页导出为 PNG。")
    return path


def new_run(kind: str) -> Path:
    parent = ROOT / "out/kimi_image_tools"
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(
        prefix=f"{datetime.now():%Y%m%d_%H%M%S}_{kind}_", dir=parent,
    ))


def api_key() -> str:
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if value:
        return value
    if SECRETS.is_file():
        if SECRETS.stat().st_mode & 0o077:
            raise ValueError("修图密钥文件权限过宽，请重新运行本机配置入口。")
        return SECRETS.read_text().strip()
    return ""


def status() -> dict:
    """Report configuration readiness without revealing credentials or using paid APIs."""
    import tomllib

    config = Path.home() / ".kimi-code/config.toml"
    configured = False
    if config.is_file():
        data = tomllib.loads(config.read_text())
        alias = data.get("default_model")
        model = data.get("models", {}).get(alias, {})
        provider = data.get("providers", {}).get(model.get("provider"), {})
        configured = bool(provider.get("api_key") or
                          os.environ.get(provider.get("api_key_env", "")))
    return {
        "kimi_credentials_configured": configured,
        "image_edit_credentials_configured": bool(api_key()),
        "image_edit_model": MODEL,
        "imagegen_cli_available": IMAGEGEN.is_file(),
        "detector_python_available": (ROOT / ".venv/bin/python").is_file(),
        "cloud_edit_live_verified": False,
        "note": "只检查配置是否存在，不证明 Key 有效、额度充足或修图质量达标。",
    }


def detect_watermark(image_path: str, use_ocr: bool = True) -> dict:
    """Run the existing local watermark detector. No upload and no model substitution."""
    source = project_file(image_path)
    run = new_run("detect")
    command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()),
               "detect-worker", "--image", str(source), "--out", str(run)]
    if not use_ocr:
        command.append("--no-ocr")
    env = dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4", WM_ORT_THREADS="4")
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        (run / "detection_error.log").write_text(result.stderr)
        raise ValueError(f"现有检测程序失败，诊断保存在 {run / 'detection_error.log'}")
    return json.loads((run / "detection.json").read_text())


def detect_worker(image_path: str, out: str, use_ocr: bool) -> None:
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from wm.contract import Thresholds
    from wm.masks import estimate_single_page

    source = project_file(image_path)
    image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
    array = np.array(image)
    mask = estimate_single_page(array, Thresholds(), use_ocr=use_ocr)
    selection = Image.fromarray(mask.binary())
    run = Path(out)
    image.save(run / "source.png")
    selection.save(run / "selection.png")
    overlay = Image.composite(Image.new("RGB", image.size, "red"), image,
                              selection.point(lambda p: 105 if p else 0))
    overlay.save(run / "selection_overlay.png")
    report = {"source": str(run / "source.png"),
              "selection": str(run / "selection.png"),
              "selection_overlay": str(run / "selection_overlay.png"),
              "detector": "wm.masks.estimate_single_page",
              "method": mask.method,
              "selected_pixels": int(np.count_nonzero(np.asarray(selection))),
              "meaning": "selection 白色为要修复区域，黑色保留；未执行修图。"}
    (run / "detection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


def selection_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as raw:
        if raw.size != size:
            raise ValueError("水印选区尺寸必须与原图完全一致。")
        if "A" in raw.getbands() and raw.getchannel("A").getextrema() != (255, 255):
            raise ValueError("请使用检测器输出的黑白 selection.png；不要传透明水印图层或叠加预览图。")
        if raw.mode not in {"1", "L", "I", "I;16"}:
            rgb = raw.convert("RGB")
            red, green, blue = rgb.split()
            if ImageChops.difference(red, green).getbbox() or ImageChops.difference(red, blue).getbbox():
                raise ValueError("选区必须为黑白图，白色表示需要修复的像素。")
        return raw.convert("L").point(lambda p: 255 if p >= 128 else 0)


def prepare_canvas(image: Image.Image, selection: Image.Image) -> tuple[Image.Image, Image.Image, tuple]:
    ratio = image.width / image.height
    target = (1536, 1024) if ratio > 1.15 else ((1024, 1536) if ratio < 0.87 else (1024, 1024))
    scale = min(target[0] / image.width, target[1] / image.height)
    dims = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    left, top = (target[0] - dims[0]) // 2, (target[1] - dims[1]) // 2
    bbox = (left, top, left + dims[0], top + dims[1])
    canvas = Image.new("RGB", target, "white")
    canvas.paste(image.resize(dims, Image.Resampling.LANCZOS), (left, top))
    edit = Image.new("L", target, 0)
    # Expand nonzero area averages back to a binary selection after resizing.
    resized = selection.resize(dims, Image.Resampling.BOX).point(lambda p: 255 if p else 0)
    edit.paste(resized, (left, top))
    api_mask = Image.new("RGBA", target, (0, 0, 0, 255))
    api_mask.putalpha(ImageOps.invert(edit))
    return canvas, api_mask, bbox


def rebuild_background(image_path: str, selection_path: str, instruction: str = "",
                       dry_run: bool = True) -> dict:
    """Rebuild masked background using a cloud image editor. dry_run is the default.

    A live call (dry_run=False) uploads this image and mask to OpenAI and uses paid
    API quota. Use an existing detector selection; do not fabricate a new selection.
    Final PNG preserves every original pixel outside the supplied selection.
    """
    source = project_file(image_path)
    mask_path = project_file(selection_path)
    image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
    selection = selection_image(mask_path, image.size)
    if not selection.getbbox():
        return {"status": "empty_selection", "api_called": False,
                "message": "现有检测没有选中水印，不执行生成，也不将此判为去水印成功。"}
    if not IMAGEGEN.is_file():
        raise ValueError("找不到 imagegen 官方脚本；请检查 KIMI_IMAGEGEN_CLI。")
    key = api_key()
    if not dry_run and not key:
        raise ValueError("尚未配置图像编辑服务 Key。Kimi/DeepSeek 对话 Key 不能代替 OpenAI 修图 Key。")
    run = new_run("preview" if dry_run else "edit")
    image.save(run / "source.png")
    selection.save(run / "selection.png")
    canvas, api_mask, bbox = prepare_canvas(image, selection)
    canvas.save(run / "model_input.png")
    api_mask.save(run / "model_mask.png")
    (run / "prompt.txt").write_text(PROMPT + "\nAdditional instruction:\n" + instruction)
    command = [str(RUNTIME), str(IMAGEGEN), "edit", "--model", MODEL,
               "--image", str(run / "model_input.png"),
               "--mask", str(run / "model_mask.png"),
               "--prompt-file", str(run / "prompt.txt"), "--no-augment",
               "--quality", "high", "--size", f"{canvas.width}x{canvas.height}",
               "--output-format", "png", "--out", str(run / "generated.png")]
    if dry_run:
        command.append("--dry-run")
    env = dict(os.environ)
    # Keep image service credentials independent of third-party chat base URLs.
    env.pop("OPENAI_BASE_URL", None)
    env.pop("OPENAI_ORG_ID", None)
    env.pop("OPENAI_PROJECT_ID", None)
    if key:
        env["OPENAI_API_KEY"] = key
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    log = result.stdout + result.stderr
    if key:
        log = log.replace(key, "[REDACTED]")
    (run / "imagegen.log").write_text(log)
    report = {"status": "dry_run" if dry_run else "needs_review", "api_called": not dry_run,
              "source": str(run / "source.png"), "selection": str(run / "selection.png"),
              "model": MODEL, "quality": "high", "run_directory": str(run),
              "original_size": list(image.size), "model_size": list(canvas.size),
              "content_box": list(bbox), "prompt": str(run / "prompt.txt"),
              "quality_verified": False,
              "note": "模型输入按比例缩放并补边。合成只保留选区内生成内容，仍需检查结构偏移与文字损伤。"}
    if result.returncode:
        report.update(status="error", message="图像服务调用失败，请查看已脱敏的 imagegen.log。")
    elif not dry_run:
        with Image.open(run / "generated.png") as generated:
            if generated.size != canvas.size:
                report.update(status="size_mismatch", message="模型输出尺寸异常，保留原始输出供检查，未自动合成。")
            else:
                import numpy as np
                from compose_semantic_repair import blend_generated_background

                patch = generated.convert("RGB").crop(bbox).resize(image.size, Image.Resampling.LANCZOS)
                merged = Image.fromarray(blend_generated_background(
                    np.asarray(image), np.asarray(patch), np.asarray(selection) >= 128))
                merged.save(run / "result.png")
                outside = ImageChops.multiply(ImageChops.difference(merged, image).convert("L"),
                                              ImageOps.invert(selection))
                report.update(result=str(run / "result.png"),
                              composition_method="generated_gradients_with_original_boundary",
                              outside_selection_unchanged=outside.getbbox() is None)
    (run / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def serve() -> None:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("watermark-image-tools", instructions=(
        "Use existing watermark detection, then call rebuild_background with its black/white selection. "
        "Always inspect status first. Dry runs do not repair images. Never report a real repair until "
        "the API has returned an image. Kimi and DeepSeek chat keys are not image editor keys. "
        "Do not read credential files or ask users to paste secrets into chat. "
        "Show source, selection and result to the user and label generated content as unverified."
    ))
    server.tool()(status)
    server.tool()(detect_watermark)
    server.tool()(rebuild_background)
    server.run(transport="stdio")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["serve", "status", "detect-worker"])
    parser.add_argument("--image")
    parser.add_argument("--out")
    parser.add_argument("--no-ocr", action="store_true")
    args = parser.parse_args()
    if args.mode == "serve":
        serve()
    elif args.mode == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    else:
        detect_worker(args.image, args.out, not args.no_ocr)
