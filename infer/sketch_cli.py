import sys
import zmq
import zlib
from PyQt5.QtWidgets import (QApplication, QWidget)
from PyQt5.QtGui import (QPainter, QPen)
from PyQt5.QtCore import Qt
import pickle as pkl

from PIL import Image, ImageDraw
import numpy as np

# Ramer-Douglas-Peucker Algorithm
from rdp import rdp


class Paint(QWidget):

    def __init__(self):
        super(Paint, self).__init__()
        self.setFixedSize(512,512)
        self.move(100, 100)
        self.setWindowTitle("sketch")
        self.setMouseTracking(False)
        self.pos_xy = []
        self.pos_xy_simplified = []

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect ("tcp://10.141.221.134:10116")

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

    def send_array(self, A, flags=0, copy=True, track=False):
        """send a numpy array with metadata"""
        md = dict(
            dtype = str(A.dtype),
            shape = A.shape,
        )
        self.socket.send_json(md, flags|zmq.SNDMORE)
        return self.socket.send(A, flags, copy=copy, track=track)

    def mouseReleaseEvent(self, event):
        #update simplified sketch
        tmp_simp = self.pos_xy_simplified
        index = len(self.pos_xy) - 1
        while(index > 1 and self.pos_xy[index] != (-1, -1)):
            index -= 1
        simp = rdp(self.pos_xy[index+1:], epsilon=1.0)
        if(len(simp) > 1):
            tmp_simp.append(simp)

        #detect the sketch
        strokes = []
        for s in tmp_simp:
            strokes.append(np.transpose(s, (1,0))/4)

        if(len(strokes) != 0):
            #print self.iu.infer(img,5)
            #self.send_array(img)
            p = pkl.dumps(strokes,2)
            z = zlib.compress(p)
            self.socket.send(z, flags=0)
            z = self.socket.recv(flags=0)
            p = zlib.decompress(z)
            new_storkes = pkl.loads(p, encoding='latin1')
            if len(new_storkes) <len(self.pos_xy_simplified):
                self.pos_xy_simplified = self.pos_xy_simplified[:len(new_storkes)]

            self.pos_xy = []
            for s in new_storkes:
                s = np.transpose(s, (1,0))*4
                ns = [list(t) for t in s] + [(-1, -1)]
                self.pos_xy += ns
            print(self.pos_xy)

        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)
        self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    pyqt_exe = Paint()
    pyqt_exe.show()
    app.exec_()
