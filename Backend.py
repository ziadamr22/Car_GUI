import sys
from termios import VEOL
from FrontEnd import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRunnable, QThreadPool
from PyQt5.QtGui import QPixmap
import cv2 as cv
import numpy as np
import serial 
global arduino
arduino = serial.Serial(port='/dev/cu.usbmodem11301'
                        ,baudrate=9600)
class sendArduino(QRunnable):
    def __init__(self,x):
        super(sendArduino,self).__init__()
        self.x = x
    def run(self):
        arduino.write(bytes(self.x,'utf-8'))
class VideoThread(QtCore.QThread):# initialise live video feed thread
    change_pixmap_signal = QtCore.pyqtSignal(np.ndarray)
    def run(self):
        # capture from web cam
        cap = cv.VideoCapture(0)
        while True:
            ret, cv_img = cap.read()
            cv_img = cv.flip(cv_img,1)
            if ret:
                self.change_pixmap_signal.emit(cv_img)
# initialise GUI Window
class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.screenshot)
        self.uplogo = QtGui.QPixmap("up.png")
        self.ui.Up.setIcon(QtGui.QIcon(self.uplogo))
        self.downlogo = QtGui.QPixmap("down.png")
        self.ui.Down.setIcon(QtGui.QIcon(self.downlogo))
        self.rightlogo = QtGui.QPixmap("right.png")
        self.ui.Right.setIcon(QtGui.QIcon(self.rightlogo))
        self.leftlogo = QtGui.QPixmap("left.png")
        self.ui.Left.setIcon(QtGui.QIcon(self.leftlogo))
        self.stoplogo = QtGui.QPixmap("stop.png")
        self.ui.Stop.setIcon(QtGui.QIcon(self.stoplogo))
        self.ui.speed1.clicked.connect(self.Speed1)
        self.ui.speed2.clicked.connect(self.Speed2)
        self.ui.speed3.clicked.connect(self.Speed3)
        self.ui.Up.clicked.connect(self.f_up)
        self.ui.Down.clicked.connect(self.f_back)
        self.ui.Right.clicked.connect(self.f_right)
        self.ui.Left.clicked.connect(self.f_left)
        self.ui.Stop.clicked.connect(self.f_stop)
        self.threadpool = QThreadPool()
        self.i=0
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update)
        self.thread.start()
    def f_up (self):
        up = sendArduino('3')
        self.threadpool.start(up)
    def f_back (self):
        back = sendArduino('4')
        self.threadpool.start(back)    
    def f_right (self):
        right = sendArduino('5')
        self.threadpool.start(right)
    def f_left (self):
        left = sendArduino('6')
        self.threadpool.start(left)
    def f_stop (self):
        stop = sendArduino('7')
        self.threadpool.start(stop)
    def Speed1 (self):
        s_1 = sendArduino('0')
        self.threadpool.start(s_1)
    def Speed2 (self):
        s_2 = sendArduino('1')
        self.threadpool.start(s_2)
    def Speed3 (self):
        s_3 = sendArduino('2')
        self.threadpool.start(s_3)
        #update video frame
    def update(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.ui.label.setPixmap(qt_img)
    #convert numpy image to qt image
    def convert_cv_qt(self, cv_img):
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data,
            w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(450, 450,
            QtCore.Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    #Screeshot function
    def screenshot(self):
        screen = QtWidgets.QApplication.primaryScreen()
        w = QtWidgets.QWidget()
        screenshot = screen.grabWindow( w.winId() )
        label = 'screenshot'+str(self.i)+'.jpg'
        screenshot.save(label,'jpg')
        self.i+=1  
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Int = Window()
    Int.show()
    sys.exit(app.exec_())