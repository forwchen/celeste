import sys
from PyQt5.QtWidgets import (QApplication, QWidget)
from PyQt5.QtGui import (QPainter, QPen)
from PyQt5.QtCore import Qt

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

    def mouseMoveEvent(self, event):
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
        # print(self.pos_xy_simplified)
        
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)
        self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    pyqt_exe = Paint()
    pyqt_exe.show()
    app.exec_()
