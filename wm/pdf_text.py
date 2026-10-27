"""Conservative removal of independently drawn translucent PDF text.

Only text-show operators are omitted. No rectangular redaction, raster repair,
or access to a clean reference is involved. Form XObjects and mixed text blocks
are deliberately left to the raster pipeline.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from io import BytesIO
import math

import numpy as np
import pymupdf as fitz
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, NameObject

_SHOW = {b"Tj", b"TJ", b"'", b'"'}


def _body_size(page):
    sizes = []
    for span in page.get_texttrace():
        if span["opacity"] > .95 and span["type"] == 0:
            sizes.extend([span["size"]] * min(len(span["chars"]), 200))
    return float(np.median(sizes)) if len(sizes) >= 40 else None


def _text(operands, op):
    parts = operands[0] if op == b"TJ" else [operands[-1]]
    return "".join(str(x) for x in parts if isinstance(x, (str, bytes)))


def strip_translucent_text(data: bytes) -> tuple[bytes, tuple[int, ...]]:
    """Return rewritten bytes and removed text-block counts per page.

    Candidate: fill-only translucent text, at least 20pt and 1.5x body size,
    rotated 15–65 degrees; horizontal candidates must repeat at least 3 times.
    These geometric checks are evidence of an overlay, not a semantic guarantee.
    """
    reader = PdfReader(BytesIO(data), strict=False)
    writer = PdfWriter()
    counts = []
    with fitz.open(stream=data, filetype="pdf") as rendered:
        for index, page in enumerate(reader.pages):
            body_size = _body_size(rendered[index])
            content = page.get_contents()
            if body_size is None or content is None:
                writer.add_page(page)
                counts.append(0)
                continue
            stream = ContentStream(content, reader)
            resources = page.get("/Resources")
            resources = resources.get_object() if resources is not None else {}
            ext = resources.get("/ExtGState", {}).get_object() if "/ExtGState" in resources else {}
            state = dict(matrix=fitz.Matrix(1, 0, 0, 1, 0, 0), size=0., alpha=1., mode=0,
                         colour=(0.,))
            stack = []
            text_matrix = fitz.Matrix(1, 0, 0, 1, 0, 0)
            blocks, block = [], None
            unsafe = False
            for position, (args, op) in enumerate(stream.operations):
                if op == b"q":
                    stack.append(deepcopy(state))
                elif op == b"Q":
                    if not stack:
                        unsafe = True
                        break
                    state = stack.pop()
                elif op == b"cm":
                    state["matrix"] = fitz.Matrix(*map(float, args)) * state["matrix"]
                elif op == b"gs":
                    gs = ext.get(args[0])
                    if gs is not None:
                        gs = gs.get_object()
                        state["alpha"] = float(gs.get("/ca", state["alpha"]))
                        # Unusual blend modes and soft masks are unsupported.
                        if gs.get("/BM", "/Normal") != "/Normal" or gs.get("/SMask", "/None") != "/None":
                            unsafe = True
                            break
                elif op == b"Tf":
                    state["size"] = float(args[1])
                elif op == b"Tr":
                    state["mode"] = int(args[0])
                elif op in (b"rg", b"g", b"k"):
                    state["colour"] = tuple(round(float(v), 4) for v in args)
                elif op in (b"cs", b"sc", b"scn"):
                    state["colour"] = None
                elif op == b"BT":
                    if block is not None:
                        unsafe = True
                        break
                    text_matrix = fitz.Matrix(1, 0, 0, 1, 0, 0)
                    block = []
                elif op == b"Tm":
                    text_matrix = fitz.Matrix(*map(float, args))
                elif op in _SHOW:
                    if block is None:
                        unsafe = True
                        break
                    matrix = text_matrix * state["matrix"]
                    physical_size = abs(state["size"]) * math.hypot(matrix.c, matrix.d)
                    angle = abs(math.degrees(math.atan2(matrix.b, matrix.a))) % 180
                    angle = min(angle, 180 - angle)
                    text = _text(args, op)
                    candidate = (state["mode"] == 0 and .10 <= state["alpha"] <= .40
                                 and physical_size >= max(20., 1.5 * body_size)
                                 and len(text.strip()) >= 6)
                    block.append((position, candidate, 15 <= angle <= 65, text,
                                  round(state["alpha"], 3), state["colour"],
                                  physical_size, (matrix.e, matrix.f)))
                elif op == b"ET":
                    if block:
                        blocks.append(block)
                    block = None
            repeats = Counter(tuple(x[3] for x in b) for b in blocks
                              if all(x[1] for x in b))
            remove = set()
            count = 0
            if not unsafe and block is None and not stack:
                # A second, horizontal line may belong to an already confirmed
                # overlay (including a different/CJK font). It is not sufficient
                # for a page to contain *any* large translucent heading: require
                # matching paint, comparable size and spatial proximity to the
                # independent rotated/repeated evidence.
                anchors = [x for b in blocks if all(x[1] for x in b)
                           and (all(x[2] for x in b) or repeats[tuple(x[3] for x in b)] >= 3)
                           for x in b]
                for b in blocks:
                    associated = bool(anchors) and all(any(
                        x[5] is not None and x[4:6] == a[4:6] and .65 <= x[6] / a[6] <= 1.5
                        and math.dist(x[7], a[7]) <= 8 * a[6]
                        for a in anchors) for x in b)
                    if all(x[1] for x in b) and (all(x[2] for x in b)
                            or repeats[tuple(x[3] for x in b)] >= 3 or associated):
                        remove.update(x[0] for x in b)
                        count += 1
            if remove:
                stream.operations = [item for n, item in enumerate(stream.operations) if n not in remove]
                page[NameObject("/Contents")] = stream
            writer.add_page(page)
            counts.append(count)
    if not any(counts):
        return data, tuple(counts)
    output = BytesIO()
    writer.write(output)
    return output.getvalue(), tuple(counts)
