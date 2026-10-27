"""Remove individual translucent image draws over body text, not shared resources.

Only top-level image invocations are supported. Small logos, opaque photographs,
background artwork and Form contents are left alone. The decision is geometric
evidence of an overlay, so results still require review without a clean reference.
"""
from io import BytesIO
from copy import deepcopy

import numpy as np
import pymupdf as fitz
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, NameObject


def strip_translucent_images(data: bytes) -> tuple[bytes, tuple[int, ...]]:
    reader = PdfReader(BytesIO(data), strict=False)
    writer = PdfWriter()
    counts = []
    with fitz.open(stream=data, filetype="pdf") as rendered:
        for index, page in enumerate(reader.pages):
            content = page.get_contents()
            resources = page.get("/Resources")
            resources = resources.get_object() if resources is not None else {}
            images = resources.get("/XObject", {})
            images = images.get_object() if hasattr(images, "get_object") else images
            states = resources.get("/ExtGState", {})
            states = states.get_object() if hasattr(states, "get_object") else states
            if content is None:
                writer.add_page(page); counts.append(0); continue
            stream = ContentStream(content, reader)
            state = dict(matrix=fitz.Matrix(1, 0, 0, 1, 0, 0), safe=True)
            stack, remove = [], set()
            pg = rendered[index]
            # Unknown page rotations and non-default page origins need a full
            # page-coordinate interpreter; do not guess their image rectangles.
            eligible = pg.rotation == 0 and pg.cropbox == pg.mediabox
            text = [(t["seqno"], fitz.Rect(t["bbox"])) for t in pg.get_texttrace()
                    if t["opacity"] > .95 and t["size"] < 20]
            image_draws = [(seq, fitz.Rect(entry[1])) for seq, entry in enumerate(pg.get_bboxlog())
                           if entry[0] == "fill-image"]
            by_xref = {i[0]: i for i in pg.get_images(full=True)}
            # Nested images require a recursive coordinate interpreter. Keep
            # them intact rather than confusing their draws with top-level ones.
            eligible &= not any(i[-1] for i in by_xref.values())
            used_draws, opaque_underlays = set(), []
            for pos, (args, op) in enumerate(stream.operations):
                if op == b"q":
                    stack.append(deepcopy(state))
                elif op == b"Q":
                    if not stack:
                        eligible = False; break
                    state = stack.pop()
                elif op == b"cm":
                    state["matrix"] = fitz.Matrix(*map(float, args)) * state["matrix"]
                elif op == b"gs":
                    gs = states.get(args[0])
                    if gs is not None:
                        gs = gs.get_object()
                        state["safe"] &= (gs.get("/BM", "/Normal") == "/Normal"
                                          and gs.get("/SMask", "/None") == "/None")
                elif op in (b"W", b"W*"):
                    state["safe"] = False
                elif op == b"Do" and eligible and state["safe"]:
                    ref = images.get(args[0])
                    if ref is None or ref.get_object().get("/Subtype") != "/Image":
                        continue
                    item = by_xref.get(ref.idnum)
                    if item is None:
                        continue
                    matrix = state["matrix"]
                    if abs(matrix.b) > .01 or abs(matrix.c) > .01:
                        continue
                    rect = fitz.Rect(0, 0, 1, 1) * matrix * pg.transformation_matrix
                    area = rect.get_area() / max(pg.rect.get_area(), 1)
                    centre = (rect.tl + rect.br) / 2
                    if not (.06 <= area <= 1.02 and .20*pg.rect.width < centre.x < .80*pg.rect.width
                            and .20*pg.rect.height < centre.y < .80*pg.rect.height):
                        continue
                    matches = [seq for seq, drawn in image_draws
                               if seq not in used_draws and max(abs(a-b) for a, b in zip(rect, drawn)) < .05]
                    if not matches:
                        continue
                    draw = min(matches); used_draws.add(draw)
                    if not item[1]:
                        if area >= .60:
                            opaque_underlays.append((draw, rect))
                        continue
                    body_text = sum(seq < draw and (rect & box).get_area() > .2*box.get_area()
                                    for seq, box in text) >= 3
                    raster_body = any(seq < draw and (rect & base).get_area() >= .9*rect.get_area()
                                      for seq, base in opaque_underlays)
                    if not (body_text or raster_body):
                        continue
                    pix = fitz.Pixmap(rendered, item[1])
                    alpha = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., 0]
                    support = alpha[alpha > 0]
                    if area*np.mean(alpha > 0) > .35:
                        continue
                    if support.size < 100 or not (20 <= np.median(support) <= 230):
                        continue
                    if np.mean(support > 235) > .01:
                        continue
                    remove.add(pos)
            if stack or not eligible:
                remove.clear()
            if remove:
                stream.operations = [entry for pos, entry in enumerate(stream.operations) if pos not in remove]
                page[NameObject("/Contents")] = stream
            writer.add_page(page)
            counts.append(len(remove))
    if not any(counts):
        return data, tuple(counts)
    output = BytesIO(); writer.write(output)
    return output.getvalue(), tuple(counts)
