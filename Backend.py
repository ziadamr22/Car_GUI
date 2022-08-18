from encodings import utf_8
import sys
from FrontEnd import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
import cv2 as cv
from PyQt5.uic import loadUi
import numpy as np
import serial 
# initialise Video Recording
class VideoThread(QtCore.QThread):
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
        self.arduino = serial.Serial(port='/dev/cu.usbmodem11301',baudrate=9600)
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


        ''' 
        8 == forward
        2 == back
        '''
        self.i=0
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
        ##sends signal to arduino
    # def ArduinoSend(self,x):
        # self.arduino.write(data=str(x,'utf-8'))
    #update video frame every thread change
    def Speed1 (self):
        self.arduino.write(bytes('0','utf-8'))
    def Speed2 (self):
        self.arduino.write(bytes('1','utf-8'))
    def Speed3 (self):
        self.arduino.write(bytes('2','utf-8'))

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        
        self.ui.label.setPixmap(qt_img)
    #convert numpy image to qt image
    def convert_cv_qt(self, cv_img):
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(450, 450, QtCore.Qt.KeepAspectRatio)
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