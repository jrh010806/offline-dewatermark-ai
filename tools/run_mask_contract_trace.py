"""Record real masks at application boundaries without changing inference.

Run this tool from a frozen source tree. The task contains source inputs only;
references belong in a separate, later scoring process. No trace is a new mask.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


class Trace:
    def __init__(self, task, folder):
        self.folder = folder
        self.threshold = task['settings']['thresholds']['mask_alpha_min']
        self.rows = {}
        self.active = None
        self.call = None
        self.tile = None
        self.index = {}
        for item in task['inputs']:
            source = np.array(Image.open(item['input']).convert('RGB'))
            key = digest(source)
            if key in self.index:
                raise ValueError('重复源图需要另行区分页次，本入口不接受含糊映射')
            self.index[key] = item

    def select(self, image):
        key = digest(image)
        item = self.index[key]
        if key not in self.rows:
            dest = self.folder / ('逐阶段_' + Path(item['input']).stem)
            dest.mkdir(exist_ok=False)
            Image.fromarray(image).save(dest / '00_实际输入.png')
            self.rows[key] = dict(id=item['id'], input_sha256=item['input_sha256'],
                                  directory=str(dest), stages=[], channels=[], lama_calls=[],
                                  reference_read=False)
        self.active = self.rows[key]
        return self.active

    def image(self, name, array):
        dest = Path(self.active['directory']) / (name + '.png')
        Image.fromarray(np.uint8(array) * 255 if array.dtype == bool else array).save(dest)
        return str(dest)

    def layer(self, name, mask, **details):
        selected = mask.binary(self.threshold) > 0
        arrays = {k: getattr(mask, k) for k in
                  ('alpha', 'footprint', 'selection', 'text_keepout', 'overlay_opacity',
                   'repeat_stroke_support', 'veil_strength', 'veil_offset')
                  if getattr(mask, k) is not None}
        np.savez_compressed(Path(self.active['directory']) / (name + '_数值.npz'), **arrays)
        row = dict(name=name, method=mask.method, repair_mode=mask.repair_mode,
                   selected_pixels=int(selected.sum()), mask_sha256=digest(selected),
                   selection=self.image(name, selected), **details)
        if mask.footprint is not None:
            row['footprint_pixels'] = int(mask.footprint.sum())
            row['footprint'] = self.image(name + '_修复权限', mask.footprint)
        self.active['stages'].append(row)

    def profile(self, frame, event, arg):
        if event != 'return' or frame.f_code.co_name != '_pixel_strokes':
            return
        values = frame.f_locals
        if 'choices' not in values:
            return
        self.image('补全_实际搜索范围_不是修复许可', values['domain'])
        self.image('补全_进入搜索范围的旧种子', values['seed_region'])
        best = values['best']
        for score, region, channel, operation in values['choices']:
            label = f'补全通道_{channel}_{operation}'
            self.active['channels'].append(dict(channel=int(channel), operation=int(operation),
                score=float(score), pixels=int(region.sum()), chosen=region is best[1],
                file=self.image(label, region)))
        self.active['channel_rule'] = dict(colour_preferred=bool(values['coloured']),
            selected_channel=int(best[2]), selected_operation=int(best[3]),
            selected_score=float(best[0]), source='recall._pixel_strokes 返回现场局部变量')

    def install(self, stack):
        from wm import lama, masks, recall, repair
        initial, estimate, expand = masks._estimate_single_page, masks.estimate_single_page, recall.expand
        prior, restore = masks.with_prior, repair.repair
        inpaint, region, tile, load = lama.inpaint, lama._inpaint_region, lama._run_tile, lama._load_session
        trace = self

        def initial_call(image, *args, **kwargs):
            self.select(image)
            result = initial(image, *args, **kwargs)
            self.layer('01_初始检测选区', result)
            return result

        def estimate_call(image, *args, **kwargs):
            self.select(image)
            result = estimate(image, *args, **kwargs)
            self.layer('02_补全后选区', result,
                       note='开关关闭时为实际跳过补全后的选区；没有模拟中间结果')
            return result

        def expand_call(image, *args, **kwargs):
            self.select(image)
            previous = sys.getprofile()
            sys.setprofile(self.profile)
            try:
                return expand(image, *args, **kwargs)
            finally:
                sys.setprofile(previous)

        def prior_call(single, shared, *args, **kwargs):
            result = prior(single, shared, *args, **kwargs)
            self.layer('03_先验策略后选区', result, prior_available=shared is not None,
                       prior_selected=result is shared,
                       note='实际 with_prior 返回；当前管线没有统一独立的保护裁剪阶段')
            return result

        def restore_call(image, mask, *args, **kwargs):
            self.select(image)
            self.layer('04_修复入口选区', mask)
            start = perf_counter()
            result, used = restore(image, mask, *args, **kwargs)
            self.active['repair_elapsed_ms'] = round((perf_counter()-start)*1000)
            self.active['repairs'] = [str(x) for x in used]
            self.image('07_实际处理结果', result)
            self.image('08_实际修改像素', np.any(image != result, axis=2))
            self.active['changed_pixels'] = int(np.any(image != result, axis=2).sum())
            self.active['outside_selected_changed'] = int((np.any(image != result, axis=2)
                & ~(mask.binary(self.threshold) > 0)).sum())
            save_json(Path(self.active['directory']) / '逐阶段记录.json', self.active)
            save_json(self.folder / '逐阶段记录索引.json', list(self.rows.values()))
            return result, used

        def inpaint_call(image, binary_mask, *args, **kwargs):
            call = dict(index=len(self.active['lama_calls']), tiles=[], input_image=image,
                        before=binary_mask.copy(), actual_union=np.zeros(image.shape[:2], bool))
            self.call = call
            self.image(f'05_LaMa调用{call["index"]}_外扩前权限', binary_mask > 0)
            self.image(f'05_LaMa调用{call["index"]}_实际源图', image)
            try:
                result = inpaint(image, binary_mask, *args, **kwargs)
                self.image(f'06_LaMa调用{call["index"]}_模型实际输入选区', call['actual_union'])
                call['actual_input_pixels'] = int(call['actual_union'].sum())
                call['lost_before_to_model'] = int((call['before'] & ~call['actual_union']).sum())
                call['added_before_to_model'] = int((~call['before'] & call['actual_union']).sum())
                return result
            finally:
                for key in ('input_image', 'before', 'actual_union'):
                    call.pop(key, None)
                self.active['lama_calls'].append(call)
                self.call = None

        def region_call(session, image, binary_mask, *args, **kwargs):
            parent = self.call['input_image']
            if not np.shares_memory(image, parent) or image.strides != parent.strides:
                raise ValueError('裁剪不是原图坐标视图，不能猜测映射')
            offset = image.__array_interface__['data'][0] - parent.__array_interface__['data'][0]
            y, remainder = divmod(offset, parent.strides[0])
            x, remainder = divmod(remainder, parent.strides[1])
            if remainder:
                raise ValueError('裁剪坐标未对齐')
            self.call['crop'] = dict(x=int(x), y=int(y), width=image.shape[1], height=image.shape[0])
            self.call['crop_mask_pixels'] = int((binary_mask > 0).sum())
            return region(session, image, binary_mask, *args, **kwargs)

        def tile_call(session, image, binary_mask):
            # These are the real loop coordinates, observed at the call site.
            frame = inspect.currentframe().f_back
            site = frame.f_locals
            x, y = int(site.get('x0', 0)), int(site.get('y0', 0))
            self.tile = dict(index=len(self.call['tiles']), x=x, y=y)
            try:
                return tile(session, image, binary_mask)
            finally:
                self.tile = None

        class Session:
            def __init__(self, session):
                self.session = session

            def run(self, names, feeds, *args, **kwargs):
                actual = feeds['mask'][0, 0] > 0
                crop = trace.call['crop']
                x, y = trace.tile['x'], trace.tile['y']
                width, height = min(actual.shape[1], crop['width']-x), min(actual.shape[0], crop['height']-y)
                if width <= 0 or height <= 0:
                    raise ValueError('模型图块坐标越界')
                trace.call['actual_union'][crop['y']+y:crop['y']+y+height,
                                           crop['x']+x:crop['x']+x+width] |= actual[:height, :width]
                stem = f'模型调用{trace.call["index"]}_图块{trace.tile["index"]:02d}'
                dest = Path(trace.active['directory']) / (stem + '_实际输入.npz')
                np.savez_compressed(dest, **feeds)
                trace.call['tiles'].append(dict(trace.tile, width=width, height=height,
                    selected_pixels=int(actual.sum()), tensor_file=str(dest),
                    mask_sha256=digest(feeds['mask'])))
                return self.session.run(names, feeds, *args, **kwargs)

        for module, name, value in ((masks, '_estimate_single_page', initial_call),
                (masks, 'estimate_single_page', estimate_call), (recall, 'expand', expand_call),
                (masks, 'with_prior', prior_call), (repair, 'repair', restore_call),
                (lama, 'inpaint', inpaint_call), (lama, '_inpaint_region', region_call),
                (lama, '_run_tile', tile_call), (lama, '_load_session', lambda: Session(load()))):
            stack.enter_context(patch.object(module, name, value))


def main():
    parser = argparse.ArgumentParser(description='真实选区逐阶段捕获；只接受源图任务')
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--candidate', action='store_true', help='显式启用隔离的深色笔画保留候选')
    parser.add_argument('--candidate-edges', action='store_true', help='显式核对原尺寸边缘对比，包含候选开关')
    parser.add_argument('--strategy-abstain', action='store_true', help='对缺少可靠反解模型的正文页面暂停生成式重绘')
    args = parser.parse_args()
    task = json.loads(args.task.read_text())
    if set(task) != {'mode', 'inputs', 'settings'}:
        parser.error('任务包含源图、模式和参数以外的字段')
    if (args.task.parent / '逐阶段记录索引.json').exists() or (args.task.parent / 'inference.json').exists():
        parser.error('已有结果，必须新建目录')
    for item in task['inputs']:
        if set(item) != {'id', 'input', 'input_sha256', 'dpi', 'page_count'} or item['page_count'] != 1:
            parser.error('当前入口只接受单页图片的源图任务')
        if hashlib.sha256(Path(item['input']).read_bytes()).hexdigest() != item['input_sha256']:
            parser.error('源图哈希不一致')
    from tools import control_worker
    trace = Trace(task, args.task.parent)
    sys.argv = [str(ROOT / 'tools/control_worker.py'), str(args.task)]
    with ExitStack() as stack:
        observations = []
        strategy = []
        if args.candidate or args.candidate_edges:
            from tools.experimental_mask_contract import enabled
            stack.enter_context(enabled(observations, edge_support=args.candidate_edges))
        if args.strategy_abstain:
            from tools.experimental_mask_strategy import enabled as guarded

            def capture(risk, note):
                trace.active['strategy'] = dict(note,
                    risk_file=trace.image('04a_策略暂缓区域_不是检测选区', risk))

            stack.enter_context(guarded(strategy, capture=capture))
        trace.install(stack)
        code = control_worker.main()
        save_json(args.task.parent / '候选源图记录.json',
                  dict(enabled=args.candidate or args.candidate_edges, edge_support=args.candidate_edges,
                       observations=observations, strategy=strategy,
                       strategy_abstain=args.strategy_abstain, reference_read=False))
        return code


if __name__ == '__main__':
    raise SystemExit(main())
