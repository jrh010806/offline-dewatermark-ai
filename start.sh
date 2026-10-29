#!/usr/bin/env bash
# 离线智能去水印 — 一键启动脚本
# 用法：
#   ./start.sh              等价于 ./start.sh test
#   ./start.sh setup        只装环境（磁盘预检 + venv + 依赖 + LaMa 模型 + 字体检查）
#   ./start.sh test         按修改影响选择日常检查；--plan 只预览范围和耗时
#   ./start.sh test-release 工程预检后完整三模式对照，供版本放行
#   ./start.sh freeze-baseline 冻结当前代码、资源、参数并运行，建立首次观察基线
#   ./start.sh control-inventory 校验全部应用输入、参考和既有固定集
#   ./start.sh control ...  防退化控制工具（resume/compare/promote 等）
#   ./start.sh check-data   校验固定回归集，不运行模型
#   ./start.sh gui [文件]   启动桌面前端
#   ./start.sh eval         仅跑固定核心回归集
#   ./start.sh report       显示最近一次最小回归报告路径
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
VENV=.venv
MIN_FREE_GB=5

say()  { printf '\033[1;34m[start]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[start]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[start]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 磁盘空间预检
check_disk() {
    local free_gb
    free_gb=$(df -g . | awk 'NR==2{print $4}')
    say "磁盘可用 ${free_gb}GB（要求 ≥${MIN_FREE_GB}GB）"
    [ "$free_gb" -ge "$MIN_FREE_GB" ] || die "磁盘空间不足 ${MIN_FREE_GB}GB，请先清理"
}

# ---------------------------------------------------------------- Python 选择
pick_python() {
    [ -n "$PY" ] && return
    for cand in python3.13 python3.12 python3.14 python3; do
        if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
    done
    [ -n "$PY" ] || die "找不到 python3"
    say "使用解释器：${PY}（$(${PY} -V 2>&1)）"
}

# ---------------------------------------------------------------- 依赖安装
setup_venv() {
    if [ ! -x "$VENV/bin/python" ]; then
        say "创建虚拟环境 $VENV"
        "$PY" -m venv "$VENV"
    fi
    if [ -f "$VENV/.audit_deps_ok" ]; then
        say "依赖已安装（跳过；删除 $VENV/.audit_deps_ok 可强制重装）"
        return
    fi
    say "安装依赖（先校验本平台能否装 requirements.lock 原样清单）"
    # lock 是 Windows/Python3.14 机生成的 freeze。本平台（如 macOS x86_64）可能装不上
    # 其中个别包（实测：onnxruntime 1.31 无 x86_64 轮、pymupdf 2.0 要求 Python>=3.14）。
    # 此时按 MIGRATION.md §5 退回手写清单 requirements.txt，仅放宽装不上的包，
    # 其余包仍按 requirements.txt 的精确钉死版本安装，装完用 eval 对表验收。
    if "$VENV/bin/pip" install --dry-run -q -r requirements.lock >/dev/null 2>&1; then
        "$VENV/bin/pip" install -r requirements.lock
    else
        warn "requirements.lock 原样装不上（平台 wheel 差异），改用 requirements.txt 并放宽 onnxruntime"
        local ort
        ort=$("$VENV/bin/pip" index versions onnxruntime 2>/dev/null \
              | awk '/LATEST|versions:/{print $NF}' | tr -d ',)' | head -1)
        [ -n "$ort" ] || die "无法确定本平台可用 onnxruntime 版本"
        grep -v '^onnxruntime==' requirements.txt > /tmp/req_local.txt
        echo "onnxruntime==$ort" >> /tmp/req_local.txt
        say "onnxruntime 放宽到 ${ort}（其余按 requirements.txt 钉死）"
        "$VENV/bin/pip" install -r /tmp/req_local.txt
    fi
    touch "$VENV/.audit_deps_ok"
}

selfcheck_deps() {
    say "依赖自检"
    "$VENV/bin/python" -c "import numpy, cv2, skimage, fitz, PIL, onnxruntime; print('  核心依赖 OK')"
    if "$VENV/bin/python" -c "import PySide6" 2>/dev/null; then
        echo "  PySide6 OK（GUI 可用）"
    else
        warn "PySide6 不可用，gui 子命令将失败"
    fi
}

# ---------------------------------------------------------------- LaMa 模型
ensure_model() {
    if "$VENV/bin/python" -c "from wm.lama import is_available; exit(0 if is_available() else 1)" 2>/dev/null; then
        say "LaMa 模型就绪"
        return
    fi
    say "首次运行：下载 LaMa ONNX 模型（~93MB，来自 HuggingFace opencv/inpainting_lama）"
    check_disk
    HF_HUB_OFFLINE=0 "$VENV/bin/python" - <<'EOF'
import numpy as np, cv2
from wm.lama import inpaint
img = np.full((512, 512, 3), 255, np.uint8)
mask = np.zeros((512, 512), np.uint8); mask[128:384, 128:384] = 255
inpaint(img, mask)  # 触发下载并做一次热身推理
print("LaMa 下载并预热完成")
EOF
}

# ---------------------------------------------------------------- 字体检查
check_fonts() {
    if "$VENV/bin/python" -c "
from wm.synth import _resolve_font
_resolve_font('hei'); _resolve_font('body'); _resolve_font('head')
" 2>/dev/null; then
        say "中文字体 OK"
        return
    fi
    # macOS：head 类缺黑体时，用系统黑体建软链到项目随包字体目录 wm/fonts/（项目在搜索路径中）
    if [ "$(uname -s)" = "Darwin" ] && [ ! -e wm/fonts/wqy-microhei.ttc ]; then
        warn "尝试用 macOS 系统字体补齐 head/body 字体软链"
        mkdir -p wm/fonts
        [ -f "/System/Library/Fonts/STHeiti Medium.ttc" ] && \
            ln -sf "/System/Library/Fonts/STHeiti Medium.ttc" wm/fonts/wqy-microhei.ttc
        [ -f "/System/Library/Fonts/Hiragino Sans GB.ttc" ] && \
            ln -sf "/System/Library/Fonts/Hiragino Sans GB.ttc" wm/fonts/wqy-zenhei.ttc
    fi
    if "$VENV/bin/python" -c "
from wm.synth import _resolve_font
_resolve_font('hei'); _resolve_font('body'); _resolve_font('head')
" 2>/dev/null; then
        say "中文字体 OK（经 wm/fonts 软链）"
    else
        warn "未找到中文字体：合成语料/评测会报错。请安装 Noto Sans CJK 或设置 WM_FONTS_DIR"
    fi
}

# ---------------------------------------------------------------- 各子命令
cmd_setup() {
    check_disk; pick_python; setup_venv; selfcheck_deps; ensure_model; check_fonts
    say "环境就绪。可以 ./start.sh test 或 ./start.sh gui"
}

cmd_eval() {
    say "固定核心回归：36 个输入文件 / 48 张或页"
    HF_HUB_OFFLINE=1 "$VENV/bin/python" tools/run_min_regression.py --suite core
}

cmd_test() {
    shift || true
    say "日常按影响分层检查；局部通过不能更新默认基线"
    HF_HUB_OFFLINE=1 QT_QPA_PLATFORM=offscreen "$VENV/bin/python" tools/daily_check.py "$@"
}

cmd_report() {
    "$VENV/bin/python" - <<'EOF'
from pathlib import Path
reports=list(Path('out/项目防退化控制').glob('*/项目防退化控制报告.html'))
reports+=list(Path('out/防退化对照').glob('*/项目防退化控制报告.html'))
reports+=list(Path('out/min_regression').glob('*/report.html'))
if not reports:
    raise SystemExit('尚无固定最小回归结果；请先执行 ./start.sh test 或 ./start.sh eval。')
print(max(reports,key=lambda p:p.stat().st_mtime_ns).resolve())
EOF
}

cmd_gui() {
    shift || true
    exec "$VENV/bin/python" -m wm.app "$@"
}

main() {
    local sub="${1:-test}"
    case "$sub" in
        setup)  cmd_setup ;;
        eval)   cmd_setup; cmd_eval ;;
        test)   cmd_test "$@" ;;
        test-release) shift; HF_HUB_OFFLINE=1 QT_QPA_PLATFORM=offscreen "$VENV/bin/python" tools/daily_check.py "$@" --level full ;;
        freeze-baseline) shift; HF_HUB_OFFLINE=1 QT_QPA_PLATFORM=offscreen "$VENV/bin/python" tools/regression_control.py freeze "$@" ;;
        control-inventory) shift; HF_HUB_OFFLINE=1 QT_QPA_PLATFORM=offscreen "$VENV/bin/python" tools/regression_control.py inventory "$@" ;;
        control) shift; HF_HUB_OFFLINE=1 QT_QPA_PLATFORM=offscreen "$VENV/bin/python" tools/regression_control.py "$@" ;;
        report) cmd_report ;;
        check-data) "$VENV/bin/python" tools/run_min_regression.py --validate-only ;;
        gui)    cmd_setup; cmd_gui "$@" ;;
        *) die "未知子命令：${sub}（可用：setup/check-data/eval/test/test-release/freeze-baseline/control-inventory/control/report/gui）" ;;
    esac
}
main "$@"
