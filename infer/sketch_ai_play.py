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

def pos_to_strokes(pos):
        ret = []
        current = []
        for i in range(len(pos)):
            if pos[i] == [-1,-1]:
                ret.append(current)
            else:
                current.append(pos[i])
        return ret
    
def strokes_to_pos(strokes):
        ret = []
        # import ipdb; ipdb.set_trace()
        for stroke in strokes:
            for i in range(len(stroke)):
                ret.append([int(stroke[i][0]), int(stroke[i][1])])
            ret.append([-1,-1])
        return ret

class SketchAI(object):
    def __init__(self, t=0.8):
        self.iur = InferUtilSketchRNN()
        self.t = 0.8
    
    def get_ith_stroke(self, i, pos):
        ret = []
        current = 0
        x0, y0 = 0,0
        for j in range(len(pos)):
            if current == i:
                if(i>1):
                    ret.append([pos[j][0] + x0, pos[j][1] + y0])
                else:
                    ret.append(pos[j])  
            elif current >= i:
                return ret
            if pos[j] == [-1,-1]:
                current += 1
                # x0, y0 = pos[j-1]
        return ret
    
    def predict_next_stroke(self, simplfied_strokes):
        # import ipdb; ipdb.set_trace()
        import random
        new_pos = strokes_to_pos(self.iur.predict(simplfied_strokes, t=random.random()))
        return self.get_ith_stroke(len(simplfied_strokes),new_pos)
        
class Paint(QWidget):
    def __init__(self, iu, players):
        super(Paint, self).__init__()
        self.config_limitUserStroke = False
        self.config_limitRNNStroke = False
        self.config_UserStrokeLimit = 100
        self.config_RNNStrokeLimit = 100
        
        self.players = players
        self.numPlayer = 2
        
        self.globalStep = 0
        
        self.setFixedSize(512,512)
        self.move(100, 100)
        self.setWindowTitle("sketch")
        self.setMouseTracking(False)
        self.pos_xy = [(238, 208), (238, 208), (238, 208), (239, 208), (239, 208), (239, 208), (240, 208), (240, 208), (240, 208), (241, 208), (241, 208), (241, 208), (241, 208), (242, 208), (242, 208), (242, 208), (242, 208), (243, 208), (243, 208), (243, 208), (244, 208), (244, 208), (245, 208), (245, 208), (246, 208), (246, 208), (246, 208), (246, 208), (246, 208), (246, 208), (-1, -1)]
        self.pos_xy_simplified = [[[238, 208], [246, 208]]]
        self.pos_xy_future = []
        self.iu = iu

        # self.slider = QSlider(Qt.Horizontal, self)
        # self.slider.setFocusPolicy(Qt.NoFocus)
        # self.slider.setGeometry(30, 40, 100, 30)
        # self.slider.valueChanged.connect(self.changeTemp)
        #
        # self.label = QLabel(self)
        # self.label.setText("Randomness")
        # self.label.setGeometry(160, 40, 100, 30)
        
        self.t = 0.0

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

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 5, Qt.SolidLine)
        painter.setPen(pen)
        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end[0] <= 0:
                    point_start = (-1, -1)
                    continue
                if point_start[0] <= 0:
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end() 

    def mouseMoveEvent(self, event):
        self.update()

    def mouseReleaseEvent(self, event):
        #player move
        playerNum = self.globalStep % self.numPlayer
        print("Player %s's turn"%(str(playerNum)))
        # import ipdb; ipdb.set_trace()
        new_pos = self.players.predict_next_stroke(self.pos_xy_simplified)
        # print(self.pos_xy_simplified)
        self.pos_xy += new_pos
        # pos_test = (-1, -1)
        # self.pos_xy.append(pos_test)
        
        #update simplified sketch
        simp = rdp(self.currentStroke(), epsilon=1.0)
        if(len(simp) > 2):
            for i in range(len(simp)):
                if(simp[i][0] <= 0):
                    simp[i] = simp[i-1]
            self.pos_xy_simplified.append(simp)
        
        #detect the sketch
        img = np.array(strokes_to_image_str(self.pos_xy_simplified))
        if(img.ndim != 0):
            print self.iu.infer(img,5)

        self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    iu = InferUtil('./ckpt/classifier/model-99001')
    alpha = SketchAI()
    pyqt_exe = Paint(iu, alpha)
    pyqt_exe.show()
    app.exec_()
