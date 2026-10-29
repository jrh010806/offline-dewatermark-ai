"""Image preview with an interactive, pixel-aligned before/after reveal."""
from __future__ import annotations

import numpy as np
import math
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget


def to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    return QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1] * 3,
                  QImage.Format_RGB888).copy()


class ImageViewer(QWidget):
    fractionChanged = Signal(float)
    rectangleFinished = Signal(object)
    brushStarted = Signal()
    brushSegment = Signal(object)
    brushFinished = Signal()
    zoomChanged = Signal(float)
    undoRequested = Signal()
    redoRequested = Signal()
    toolRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("处理前后对比画布")
        self.setAccessibleDescription("左右拖动分隔线查看原图和结果；方向键微调，双击居中")
        self.before = QPixmap()
        self.after = QPixmap()
        self.fraction = 0.5
        self.comparing = False
        self.caption = "原图"
        self._dragging = False
        self._page_key = None
        self.selection_mode = False
        self.search_rectangle = None
        self._selection_start = None
        self.zoom = 1.0
        self.pan = QPointF()
        self.brush_mode = False
        self.brush_include = True
        self.brush_diameter = 12
        self.magnifier_enabled = True
        self._overlay = QPixmap()
        self._hide_overlay = False
        self._hover = None
        self._painting = False
        self._last_brush = None
        self._pan_start = None
        self._space_down = False
        self.setMouseTracking(True)

    def set_brush_mode(self, enabled, *, include=True):
        self.finish_brush()
        self.brush_mode = bool(enabled)
        self.brush_include = include
        self._space_down = False
        self._pan_start = None
        self._hide_overlay = False
        self.setCursor(Qt.CrossCursor if enabled else
                       Qt.SplitHCursor if self.comparing else Qt.ArrowCursor)
        self.update()

    def set_selection_overlay(self, selected=None):
        if selected is None:
            self._overlay = QPixmap()
        else:
            rgba = np.zeros((*selected.shape, 4), np.uint8)
            rgba[..., :3] = [255, 40, 40]
            rgba[..., 3] = np.where(selected, 105, 0)
            self._overlay = QPixmap.fromImage(QImage(
                rgba.data, rgba.shape[1], rgba.shape[0], rgba.shape[1]*4,
                QImage.Format_RGBA8888).copy())
        self.update()

    def update_selection_overlay(self, selected, box):
        if self._overlay.isNull():
            self.set_selection_overlay(selected)
            return
        x0, y0, x1, y1 = box
        rgba = np.zeros((y1-y0, x1-x0, 4), np.uint8)
        rgba[..., :3] = [255, 40, 40]
        rgba[..., 3] = np.where(selected[y0:y1, x0:x1], 105, 0)
        patch = QImage(rgba.data, x1-x0, y1-y0, (x1-x0)*4, QImage.Format_RGBA8888)
        painter = QPainter(self._overlay)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.drawImage(QPointF(x0, y0), patch)
        painter.end()
        self.update()

    def finish_brush(self):
        if self._painting:
            self._painting = False
            self._last_brush = None
            self.brushFinished.emit()

    def fit_image(self):
        self.zoom = 1.0
        self.pan = QPointF()
        self.zoomChanged.emit(self.image_scale())
        self.update()

    def image_scale(self):
        return self.image_rect().width()/self.before.width() if not self.before.isNull() else 1.

    def zoom_at(self, point, factor):
        if self.before.isNull():
            return
        self.finish_brush()
        rect = self.image_rect()
        x = (point.x()-rect.left())/rect.width()
        y = (point.y()-rect.top())/rect.height()
        new_zoom = max(.25, min(128., self.zoom*factor))
        self.zoom = new_zoom
        new_rect = self.image_rect()
        self.pan += QPointF(point.x()-(new_rect.left()+x*new_rect.width()),
                            point.y()-(new_rect.top()+y*new_rect.height()))
        self.zoomChanged.emit(self.image_scale())
        self.update()

    def _paint_segment(self, position):
        # Keep off-image motion outside the image. Clamping each event would
        # paint a border when a held brush is moved through the letterbox.
        rect = self.image_rect()
        point = ((position.x()-rect.left())*self.before.width()/rect.width(),
                 (position.y()-rect.top())*self.before.height()/rect.height())
        self.brushSegment.emit((self._last_brush or point, point,
                                self.brush_diameter, self.brush_include))
        self._last_brush = point

    def set_selection_mode(self, enabled):
        self.selection_mode = bool(enabled)
        self._selection_start = None
        self._dragging = False
        self.setCursor(Qt.CrossCursor if enabled else
                       Qt.SplitHCursor if self.comparing else Qt.ArrowCursor)
        self.update()

    def clear_search_rectangle(self):
        self.search_rectangle = None
        self._selection_start = None
        self.update()

    def _image_point(self, point):
        rect = self.image_rect()
        return (max(0., min(float(self.before.width()),
                            (point.x()-rect.left()) * self.before.width()/rect.width())),
                max(0., min(float(self.before.height()),
                            (point.y()-rect.top()) * self.before.height()/rect.height())))

    def _move_rectangle(self, point):
        x, y = self._image_point(point)
        sx, sy = self._selection_start
        self.search_rectangle = (math.floor(min(x, sx)), math.floor(min(y, sy)),
                                 math.ceil(max(x, sx)), math.ceil(max(y, sy)))
        self.update()

    def set_images(self, before, after=None, *, page_key=None, caption="原图"):
        if page_key != self._page_key:
            self.finish_brush()
            self.zoom = 1.0
            self.pan = QPointF()
            self._overlay = QPixmap()
            self.set_brush_mode(False)
            self.set_fraction(0.5)
            self.clear_search_rectangle()
            self.set_selection_mode(False)
        self._page_key = page_key
        self.before = QPixmap.fromImage(to_qimage(before))
        self.after = QPixmap.fromImage(to_qimage(after)) if after is not None else QPixmap()
        self.comparing = after is not None
        self.caption = caption
        self._dragging = False
        self.setCursor(Qt.CrossCursor if self.selection_mode or self.brush_mode else
                       Qt.SplitHCursor if self.comparing else Qt.ArrowCursor)
        self.zoomChanged.emit(self.image_scale())
        self.update()

    def clear(self):
        self.finish_brush()
        self.set_brush_mode(False)
        self.set_selection_overlay()
        self.before = QPixmap()
        self.after = QPixmap()
        self.comparing = False
        self._dragging = False
        self._page_key = None
        self.set_selection_mode(False)
        self.clear_search_rectangle()
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def image_rect(self) -> QRectF:
        if self.before.isNull():
            return QRectF()
        available = QRectF(self.rect()).adjusted(28, 64, -28, -28)
        scale = min(available.width() / self.before.width(),
                    available.height() / self.before.height())
        width, height = self.before.width() * scale * self.zoom, self.before.height() * scale * self.zoom
        return QRectF(available.center().x() - width / 2 + self.pan.x(),
                      available.center().y() - height / 2 + self.pan.y(), width, height)

    def set_fraction(self, fraction):
        fraction = max(0.0, min(1.0, float(fraction)))
        if fraction != self.fraction:
            self.fraction = fraction
            self.fractionChanged.emit(fraction)
            self.update()

    def _move_divider(self, x):
        rect = self.image_rect()
        if rect.width() > 0:
            self.set_fraction((x - rect.left()) / rect.width())

    def mousePressEvent(self, event):
        self._hover = event.position()
        if (event.button() == Qt.MiddleButton or
                (event.button() == Qt.LeftButton and self._space_down)) and not self.before.isNull():
            self.finish_brush()
            self.setFocus()
            self._pan_start = (event.position(), QPointF(self.pan))
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        elif self.brush_mode and event.button() == Qt.LeftButton:
            if self.image_rect().contains(event.position()):
                self.setFocus()
                self._painting = True
                self._last_brush = None
                self.brushStarted.emit()
                self._paint_segment(event.position())
            event.accept()
        elif self.selection_mode and event.button() == Qt.LeftButton:
            if not self.before.isNull() and self.image_rect().contains(event.position()):
                self.setFocus()
                self._selection_start = self._image_point(event.position())
                self._move_rectangle(event.position())
            event.accept()
        elif self.comparing and event.button() == Qt.LeftButton:
            self.setFocus()
            self._dragging = True
            self._move_divider(event.position().x())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self._hover = event.position()
        if self._pan_start is not None:
            position, initial = self._pan_start
            self.pan = initial + event.position()-position
            self.update()
            event.accept()
        elif self._painting:
            self._paint_segment(event.position())
            self.update()
            event.accept()
        elif self._selection_start is not None:
            self._move_rectangle(event.position())
            event.accept()
        elif self._dragging:
            self._move_divider(event.position().x())
            event.accept()
        elif self.brush_mode:
            self.update()

    def mouseReleaseEvent(self, event):
        if self._pan_start is not None:
            self._pan_start = None
            self.setCursor(Qt.OpenHandCursor if self._space_down else
                           Qt.CrossCursor if self.brush_mode else
                           Qt.SplitHCursor if self.comparing else Qt.ArrowCursor)
            event.accept()
        elif self._painting and event.button() == Qt.LeftButton:
            self._paint_segment(event.position())
            self.finish_brush()
            event.accept()
        elif self._selection_start is not None and event.button() == Qt.LeftButton:
            self._move_rectangle(event.position())
            self._selection_start = None
            self.rectangleFinished.emit(self.search_rectangle)
            event.accept()
        elif self._dragging and event.button() == Qt.LeftButton:
            self._move_divider(event.position().x())
            self._dragging = False
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if not self.selection_mode and not self.brush_mode and self.comparing and event.button() == Qt.LeftButton:
            self.set_fraction(0.5)
            self._dragging = False
            event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.finish_brush()
            self._space_down = True
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        if self.brush_mode:
            command = event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)
            if event.key() == Qt.Key_H:
                self._hide_overlay = True
                self.update()
            elif command and event.key() in (Qt.Key_Z, Qt.Key_Y):
                self.finish_brush()
                if event.key() == Qt.Key_Y or event.modifiers() & Qt.ShiftModifier:
                    self.redoRequested.emit()
                else:
                    self.undoRequested.emit()
            elif event.key() in (Qt.Key_B, Qt.Key_E):
                self.toolRequested.emit('add' if event.key() == Qt.Key_B else 'erase')
            elif event.key() == Qt.Key_Escape:
                self.finish_brush()
            event.accept()
            return
        if self.selection_mode:
            if event.key() == Qt.Key_Escape:
                self.clear_search_rectangle()
            event.accept()
            return
        if self.comparing:
            values = {Qt.Key_Left: self.fraction - 0.02,
                      Qt.Key_Right: self.fraction + 0.02,
                      Qt.Key_Home: 0, Qt.Key_End: 1}
            if event.key() in values:
                self.set_fraction(values[event.key()])
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            self._pan_start = None
            self.setCursor(Qt.CrossCursor if self.brush_mode else
                           Qt.SplitHCursor if self.comparing else Qt.ArrowCursor)
        if event.key() == Qt.Key_H:
            self._hide_overlay = False
            self.update()
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self.finish_brush()
        self._hide_overlay = self._space_down = False
        self._pan_start = None
        self.update()
        super().focusOutEvent(event)

    def leaveEvent(self, event):
        if not self._painting:
            self._hover = None
            self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if not delta:
            delta = event.pixelDelta().y()
        if delta:
            self.zoom_at(event.position(), 1.2 if delta > 0 else 1/1.2)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.zoomChanged.emit(self.image_scale())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#111516"))
        painter.drawRoundedRect(QRectF(self.rect()), 16, 16)
        if self.before.isNull():
            painter.setPen(QColor("#edf2f1"))
            font = painter.font()
            font.setPointSize(18)
            painter.setFont(font)
            painter.drawText(self.rect().adjusted(20, -24, -20, -24), Qt.AlignCenter,
                             "导入图片，开始去水印")
            font.setPointSize(11)
            painter.setFont(font)
            painter.setPen(QColor("#929f9b"))
            painter.drawText(self.rect().adjusted(20, 50, -20, 50), Qt.AlignCenter,
                             "处理完成后，左右拖动查看前后变化")
            return
        rect = self.image_rect()
        painter.save()
        painter.setClipRect(QRectF(self.rect()).adjusted(4, 54, -4, -4))
        painter.setRenderHint(QPainter.SmoothPixmapTransform, self.image_scale() < 1)
        painter.drawPixmap(rect, self.after if self.comparing else self.before,
                           QRectF(self.before.rect()))
        if self.comparing:
            cut = rect.left() + rect.width() * self.fraction
            painter.save()
            painter.setClipRect(QRectF(rect.left(), rect.top(), cut - rect.left(), rect.height()))
            painter.drawPixmap(rect, self.before, QRectF(self.before.rect()))
            painter.restore()
            painter.setPen(QPen(QColor(255, 255, 255, 210), 2))
            painter.drawLine(QPointF(cut, rect.top()), QPointF(cut, rect.bottom()))
            center = QPointF(cut, rect.center().y())
            painter.setBrush(QColor(35, 43, 41, 210))
            painter.drawEllipse(center, 24, 24)
            painter.setPen(QPen(QColor("#ffffff"), 3, Qt.SolidLine, Qt.RoundCap))
            for sign in (-1, 1):
                x, y = center.x(), center.y()
                painter.drawLine(QPointF(x + sign * 6, y - 7), QPointF(x + sign * 13, y))
                painter.drawLine(QPointF(x + sign * 13, y), QPointF(x + sign * 6, y + 7))
        if not self._overlay.isNull() and not self._hide_overlay:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
            painter.drawPixmap(rect, self._overlay, QRectF(self._overlay.rect()))
        if self.search_rectangle is not None:
            x0, y0, x1, y1 = self.search_rectangle
            box = QRectF(rect.left()+x0*rect.width()/self.before.width(),
                         rect.top()+y0*rect.height()/self.before.height(),
                         (x1-x0)*rect.width()/self.before.width(),
                         (y1-y0)*rect.height()/self.before.height())
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor('#ffd17a'), 2, Qt.DashLine))
            painter.drawRect(box)
        if self.brush_mode and self._hover is not None and rect.contains(self._hover):
            radius = self.brush_diameter*self.image_scale()/2
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor('white'), 2))
            painter.drawEllipse(self._hover, radius, radius)
            painter.setPen(QPen(QColor('#00875d') if self.brush_include else QColor('#ffbf69'), 1))
            painter.drawEllipse(self._hover, radius+2, radius+2)
            if self.magnifier_enabled and self.width() >= 240:
                self._paint_magnifier(painter)
        painter.restore()
        labels = [(QRectF(20, 18, 145, 28), "搜索范围（非修复选区）" if self.selection_mode else
                   "处理前" if self.comparing else self.caption)]
        if self.comparing:
            labels.append((QRectF(self.width() - 102, 18, 82, 28), "处理后"))
        for badge, text in labels:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#262e2c"))
            painter.drawRoundedRect(badge, 8, 8)
            painter.setPen(QColor("#edf2f1"))
            painter.drawText(badge, Qt.AlignCenter, text)

    def _paint_magnifier(self, painter):
        """Use original image pixels, with a pixel-aligned overlay and brush."""
        side, magnification = 160., 4.
        x, y = self._image_point(self._hover)
        left = self._hover.x()+26
        top = self._hover.y()+26
        if left+side+8 > self.width():
            left = self._hover.x()-side-26
        if top+side+32 > self.height():
            top = self._hover.y()-side-50
        left = max(8., min(left, self.width()-side-8))
        top = max(56., min(top, self.height()-side-32))
        target = QRectF(left, top, side, side)
        span = side/magnification
        source = QRectF(x-span/2, y-span/2, span, span)
        painter.save()
        painter.setClipRect(target)
        painter.fillRect(target, QColor('#26322e'))
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        # Draw the full pixmap at 4x so crops at the image edge do not stretch.
        full = QRectF(left-source.x()*magnification, top-source.y()*magnification,
                      self.before.width()*magnification, self.before.height()*magnification)
        painter.drawPixmap(full, self.before, QRectF(self.before.rect()))
        if not self._overlay.isNull() and not self._hide_overlay:
            painter.drawPixmap(full, self._overlay, QRectF(self._overlay.rect()))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor('white'), 1))
        radius = self.brush_diameter*magnification/2
        painter.drawEllipse(target.center(), radius, radius)
        c = target.center()
        painter.drawLine(c-QPointF(5, 0), c+QPointF(5, 0))
        painter.drawLine(c-QPointF(0, 5), c+QPointF(0, 5))
        painter.restore()
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor('#d6eee3'), 2))
        painter.drawRect(target)
        painter.fillRect(QRectF(left, top+side, side, 24), QColor('#26322e'))
        painter.setPen(QColor('white'))
        painter.drawText(QRectF(left, top+side, side, 24), Qt.AlignCenter,
                         '原像素 4× · ' + ('添加' if self.brush_include else '擦除'))
