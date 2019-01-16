import sys
from PyQt5.QtWidgets import (QApplication, QWidget)
from PyQt5.QtGui import (QPainter, QPen)
from PyQt5.QtCore import Qt

from PIL import Image, ImageDraw
import numpy as np

# Ramer-Douglas-Peucker Algorithm
from rdp import rdp

from infer import InferUtil

def strokes_to_image_str(strokes_):
    if(strokes_ is None or len(strokes_) == 0):
        return
    _tmp = np.concatenate(strokes_, axis=0)

    lower = np.min(_tmp[:, 0:2], axis=0)
    strokes = []
    for s in strokes_:
        t = np.array(s)-lower
        t = np.transpose(t, (1,0))
        strokes.append(t)
    strokes = map(lambda x: x/2, strokes)
    print(strokes)

    image = Image.new("P", (256,256), color=255)
    image_draw = ImageDraw.Draw(image)
    for stroke in strokes:
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i],
                             stroke[1][i],
                             stroke[0][i+1],
                             stroke[1][i+1]],
                             fill=0, width=5)
    img_size = 128
    image = image.resize((img_size,img_size))
    return image

class Paint(QWidget):

    def __init__(self, iu):
        super(Paint, self).__init__()
        self.setFixedSize(512,512)
        self.move(100, 100)
        self.setWindowTitle("sketch")
        self.setMouseTracking(False)
        self.pos_xy = []
        self.pos_xy_simplified = []
        self.iu = iu

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 5, Qt.SolidLine)
        painter.setPen(pen)
        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    def mouseMoveEvent (self, event):
        pos_tmp = (event.pos().x(), event.pos().y())
        self.pos_xy.append(pos_tmp)
        self.update()

    def mouseReleaseEvent(self, event):
        #update simplified sketch
        index = len(self.pos_xy) - 1
        while(index > 1 and self.pos_xy[index] != (-1, -1)):
            index -= 1
        simp = rdp(self.pos_xy[index+1:], epsilon=1.0)
        if(len(simp) > 1):
            self.pos_xy_simplified.append(simp)

        #detect the sketch
        #img = np.array(Image.open('bee.png'))
        img = np.array(strokes_to_image_str(self.pos_xy_simplified))
        if(img.ndim != 0):
            print self.iu.infer(img,5)

        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)
        self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    iu = InferUtil('./ckpt/model-99001')
    pyqt_exe = Paint(iu)
    pyqt_exe.show()
    app.exec_()
