"""Image preview with an interactive, pixel-aligned before/after reveal."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget


def to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    return QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1] * 3,
                  QImage.Format_RGB888).copy()


class ImageViewer(QWidget):
    fractionChanged = Signal(float)

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

    def set_images(self, before, after=None, *, page_key=None, caption="原图"):
        if page_key != self._page_key:
            self.set_fraction(0.5)
        self._page_key = page_key
        self.before = QPixmap.fromImage(to_qimage(before))
        self.after = QPixmap.fromImage(to_qimage(after)) if after is not None else QPixmap()
        self.comparing = after is not None
        self.caption = caption
        self._dragging = False
        self.setCursor(Qt.SplitHCursor if self.comparing else Qt.ArrowCursor)
        self.update()

    def clear(self):
        self.before = QPixmap()
        self.after = QPixmap()
        self.comparing = False
        self._dragging = False
        self._page_key = None
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def image_rect(self) -> QRectF:
        if self.before.isNull():
            return QRectF()
        available = QRectF(self.rect()).adjusted(28, 64, -28, -28)
        scale = min(available.width() / self.before.width(),
                    available.height() / self.before.height())
        width, height = self.before.width() * scale, self.before.height() * scale
        return QRectF(available.center().x() - width / 2,
                      available.center().y() - height / 2, width, height)

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
        if self.comparing and event.button() == Qt.LeftButton:
            self.setFocus()
            self._dragging = True
            self._move_divider(event.position().x())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._move_divider(event.position().x())
            event.accept()

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.LeftButton:
            self._move_divider(event.position().x())
            self._dragging = False
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if self.comparing and event.button() == Qt.LeftButton:
            self.set_fraction(0.5)
            self._dragging = False
            event.accept()

    def keyPressEvent(self, event):
        if self.comparing:
            values = {Qt.Key_Left: self.fraction - 0.02,
                      Qt.Key_Right: self.fraction + 0.02,
                      Qt.Key_Home: 0, Qt.Key_End: 1}
            if event.key() in values:
                self.set_fraction(values[event.key()])
                event.accept()
                return
        super().keyPressEvent(event)

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
        labels = [(QRectF(20, 18, 82, 28), "处理前" if self.comparing else self.caption)]
        if self.comparing:
            labels.append((QRectF(self.width() - 102, 18, 82, 28), "处理后"))
        for badge, text in labels:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#262e2c"))
            painter.drawRoundedRect(badge, 8, 8)
            painter.setPen(QColor("#edf2f1"))
            painter.drawText(badge, Qt.AlignCenter, text)
