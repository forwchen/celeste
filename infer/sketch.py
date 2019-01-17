import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QSlider, QLabel)
from PyQt5.QtGui import (QPainter, QPen)
from PyQt5.QtCore import Qt

from PIL import Image, ImageDraw
import numpy as np

# Ramer-Douglas-Peucker Algorithm
from rdp import rdp

from infer import InferUtil
from infer_rnn import InferUtilSketchRNN

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

    def __init__(self, iu, iur):
        super(Paint, self).__init__()
        self.config_limitUserStroke = False
        self.config_limitRNNStroke = False
        self.config_UserStrokeLimit = 100
        self.config_RNNStrokeLimit = 100
        
        self.setFixedSize(512,512)
        self.move(100, 100)
        self.setWindowTitle("sketch")
        self.setMouseTracking(False)
        self.pos_xy = []
        self.pos_xy_simplified = []
        self.pos_xy_future = []
        self.iu = iu
        self.iur = iur
        

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.setGeometry(30, 40, 100, 30)
        self.slider.valueChanged.connect(self.changeTemp)
        
        self.label = QLabel(self)
        self.label.setText("Randomness")
        self.label.setGeometry(160, 40, 100, 30)
        
        self.t = 0.0

    def changeTemp(self, value):
        self.t = self.slider.value() / 100.0
        self.pos_xy_future = self.strokes_to_pos(iur.predict(self.pos_xy_simplified))
        self.update()
        

    def currentStroke(self):
        index = len(self.pos_xy) - 1
        while(index > 1 and self.pos_xy[index] != (-1, -1)):
            index -= 1
        return self.pos_xy[index+1:]
        
    def strokeLength(self, stroke):
        ret = 0
        if(len(stroke) < 2):
            return ret
        x0, y0 = stroke[0]
        for x, y in stroke:
            ret += np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            x0, y0 = x, y
        return ret
        
    def pos_to_strokes(self, pos):
        ret = []
        current = []
        for i in range(len(pos)):
            if pos[i] == [-1,-1]:
                ret.append(current)
            else:
                current.append(pos[i])
        return ret
    
    def strokes_to_pos(self, strokes):
        ret = []
        # import ipdb; ipdb.set_trace()
        for stroke in strokes:
            for i in range(len(stroke)):
                ret.append([int(stroke[i][0]), int(stroke[i][1])])
            ret.append([-1,-1])
        return ret

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
        
        pen = QPen(Qt.blue, 2, Qt.DotLine)
        painter.setPen(pen) 
        strokeLength = 0  

        if len(self.pos_xy_future) > 1:
            point_start = self.pos_xy_future[0]
            for pos_tmp in self.pos_xy_future:
                point_end = pos_tmp

                if point_end[0] == -1:
                    point_start = (-1, -1)
                    continue
                if point_start[0] == -1:
                    point_start = point_end
                    continue
                if point_end[0] == 0 or point_start[0] == 0:
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                strokeLength += np.sqrt((point_start[0] - point_end[0]) ** 2 + (point_start[1] - point_end[1]) ** 2)
                if(self.config_limitRNNStroke and strokeLength > self.config_RNNStrokeLimit):
                    break
                point_start = point_end
                
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
        #limit on length
        if(self.config_limitUserStroke and self.strokeLength(self.currentStroke()) > self.config_UserStrokeLimit):
            print('overLimit')
        else:
            pos_tmp = (event.pos().x(), event.pos().y())
            self.pos_xy.append(pos_tmp)
            self.update()

    def mouseReleaseEvent(self, event):
        #update simplified sketch
        simp = rdp(self.currentStroke(), epsilon=1.0)
        if(len(simp) > 1):
            self.pos_xy_simplified.append(simp)

        #detect the sketch
        img = np.array(strokes_to_image_str(self.pos_xy_simplified))
        if(img.ndim != 0):
            print self.iu.infer(img,5)
            
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)
        
        #prefict the future
        self.pos_xy_future = self.strokes_to_pos(iur.predict(self.pos_xy_simplified))

        self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    iu = InferUtil('./ckpt/classifier/model-99001')
    iur = InferUtilSketchRNN()
    pyqt_exe = Paint(iu, iur)
    pyqt_exe.show()
    app.exec_()
