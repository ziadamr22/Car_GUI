import sys
import imutils
from StitchUI import Ui_Dialog
from FrontEnd import Ui_MainWindow_
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRunnable, QThreadPool
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog,QFileDialog
import cv2 as cv
import numpy as np
import serial 
import matplotlib.pyplot as plt
import glob
global arduino
#Location of Bluetooth Module port
port = '/dev/cu.usbmodem11301'
# port = '/dev/cu.HC-06'
arduino = serial.Serial(port= port
                        ,baudrate=9600,timeout=1)
class getArduino(QRunnable):
    def __init__(self):
        super(getArduino,self).__init__()
        self.signal = None
    def run(self):
            self.signal =   arduino.readline().strip().decode("utf-8")
    def getRead(self):
        Ar_signal = self.signal
        return  Ar_signal

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

#initialise stitching window
class StitchUI(QDialog):
    def __init__(self,parent =None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.location = ""
        self.ui.Choose.clicked.connect(self.dialog)
        self.ui.Initiate.clicked.connect(self.MAINF)
    def dialog(self):
        self.location = self.getDirectory()
        self.ui.label.setText(self.location)
    def getDirectory(self):
        response = QFileDialog.getExistingDirectory(
            self,
            caption = "Select a folder",
        )
        return response
    def homography_stitching(self,keypoints_train_img, keypoints_query_img, matches, reprojThresh):
        keypoints_train_img = np.float32([keypoint.pt for keypoint in keypoints_train_img])
        keypoints_query_img = np.float32([keypoint.pt for keypoint in keypoints_query_img])

        if len(matches) >4:
            points_train = np.float32([keypoints_train_img[m.queryIdx] for m in matches])
            points_query = np.float32([keypoints_query_img[m.trainIdx] for m in matches])
            (H, status) = cv.findHomography(points_train, points_query, cv.RANSAC, reprojThresh)
            return (matches, H, status)
        else:
            print("fail")
            return None
    def _get_kp_features(self,image):

        (keypoints, features) = cv.SIFT_create().detectAndCompute(image, None)

        return (keypoints, features)
    def create_matching_object(self,method, crossCheck):
        if method == 'sift' or method == 'surf':
            bf = cv.BFMatcher(cv.NORM_L2, crossCheck=crossCheck)
        elif method == 'orb' or method == 'brisk':
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=crossCheck)
        return bf

    def key_points_matching_KNN(self,features_train_img, features_query_img, ratio, method):
        bf = self.create_matching_object(method, crossCheck=False)
        rawMatches = bf.knnMatch(features_train_img, features_query_img, k=2)
        print("Raw matches (knn):", len(rawMatches))
        matches = []
        for m,n in rawMatches:
            if m.distance < n.distance * ratio:
                matches.append(m)
        return matches


    def _adjust(self,StitchedImage):
        gray = cv.cvtColor(StitchedImage, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv.boundingRect(cnt)
        StitchedImage = StitchedImage[y:y + h, x:x + w]
        return StitchedImage

    def _stitcher(self,query_photo, train_photo):
        query_photo_gray = cv.cvtColor(query_photo, cv.COLOR_RGB2GRAY)
        train_photo_gray = cv.cvtColor(train_photo, cv.COLOR_RGB2GRAY)
        keypoints_train_img, features_train_img = self._get_kp_features(train_photo_gray)
        keypoints_query_img, features_query_img = self._get_kp_features(query_photo_gray)

        if self.feature_to_match == 'knn':
            matches = self.key_points_matching_KNN(features_train_img, features_query_img, ratio=0.75,
                                            method=self.feature_extraction_algo)
        M = self.homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh=54)
        if M is None:
            print("Error!")
            return
        (matches, Homography_Matrix, status) = M
        width = query_photo.shape[1] + train_photo.shape[1]
        height = max(query_photo.shape[0], train_photo.shape[0])
        result = cv.warpPerspective(train_photo, Homography_Matrix, (width, height))
        result[0:query_photo.shape[0], 0:query_photo.shape[1]] = query_photo

        return result
    def MAINF(self):
        if __name__ == "__main__":
            self.feature_extraction_algo = 'sift'
            self.feature_to_match = 'knn'
            temp = self.location+'/*.jpg'
            image_paths = glob.glob(temp)
            index = -1
            images = []
            for image in image_paths:
                img = cv.imread(image)
                images.append(img)
            for i in range(0,len(images)):
                index=index+1
                maxmatches=0
                for j in range(0,len(images)):
                    self.keypoints_i ,features_i=self._get_kp_features(images[index])
                    self.keypoints_j,features_j = self._get_kp_features(images[j])
                    matches = self.key_points_matching_KNN(features_i, features_j, ratio=0.75,method=self.feature_extraction_algo)
                    if len(matches)>maxmatches and index!=j:
                        maxmatches = len(matches)
                        pair=(index,j)
                if len(images)>1:
                    StitchedImage = self._stitcher(images[pair[1]], images[pair[0]])
                    StitchedImage=self._adjust(StitchedImage)
                    if StitchedImage.shape[1]==(images[pair[0]].shape[1])or StitchedImage.shape[1]==(images[pair[1]].shape[1]):
                            StitchedImage =self._stitcher(images[pair[0]], images[pair[1]])
                            StitchedImage = self._adjust(StitchedImage)
                    images.pop(pair[0])
                    images.pop(pair[1]-1)
                    index=index-1
                    images.insert(0,StitchedImage)
            cv.imwrite("Stitched.jpg", StitchedImage)
            image = StitchedImage
            original = image.copy()
            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

            hsv_lower = np.array([23, 19, 141])
            hsv_upper = np.array([167, 255, 255])
            mask = cv.inRange(hsv, hsv_lower, hsv_upper)
            result = cv.bitwise_and(original, original, mask=mask)

            img = result
            img = cv.GaussianBlur(img, (7, 7), 1)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            thresh, thresh_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)
            rgb_img = cv.cvtColor(thresh_img, cv.COLOR_BGR2RGB)
            plt.imshow(rgb_img)
            plt.show()
            conts = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            conts = imutils.grab_contours(conts)
            print(len(conts))
            cont_img = np.zeros(img.shape)
            for c in conts:
                box = cv.minAreaRect(c)
                box = cv.boxPoints(box)
                box = np.array(box, dtype='int')
                x, y, w, h = cv.boundingRect(c)
                aspect_ratio = float(w) / h
                if cv.contourArea(c) < 10000 or w > 1200 or h > 430:
                    continue
                print(x, y, w, h)
                cv.drawContours(cont_img, [c], -1, (0, 255, 255, 2))
                cv.drawContours(cont_img, [box], -1, (255, 255, 255, 1))
                approx = cv.approxPolyDP(c, 0.03 * cv.arcLength(c, True), True)
                print(len(approx))
                if len(approx) >= 5 and aspect_ratio < 1.2:
                    if w > 10 and h > 10:
                        cv.drawContours(image, [box], -1, (255, 179, 0,), 10)
                        cv.putText(image, "Sea Star", (x - 50, y - 20), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
                elif len(approx) >=5 and len(approx) < 8 and aspect_ratio > 1.2:
                    if w > 10 and h > 10 and w < 700 and h < 700:
                        cv.putText(image, "Sponge", (x - 20, y - 20), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
                        cv.drawContours(image, [box], -1, (15, 242, 113,), 10)

                elif len(approx) == 4 and aspect_ratio < 1.5:
                    cv.putText(image, "Coral Fragment", (x - 100, y - 20), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
                    cv.drawContours(image, [box], -1, (0, 68, 255,), 10)
                elif aspect_ratio > 3:
                    cv.putText(image, "Coral Colony", (x + 100, y - 20), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
                    cv.drawContours(image, [box], -1, (12, 121, 59,), 10)
            cv.imwrite('final.jpg', image)
            self.close()
            

# initialise GUI Window
class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow_()
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
        self.ui.swTask.clicked.connect(self.Stitch)
        self.ui.MOTOR.clicked.connect(self.selectMotor)
        self.ui.ESC.clicked.connect(self.selectESC)
        self.read = getArduino()
        self.threadpool = QThreadPool()
        self.i=0
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update)
        self.thread.start()
    def Stitch(self):
        stitch = StitchUI(self)
        stitch.exec()
    # def showVolt(self):
    #     volt = getArduino()
    #     volt.run()
    #     read = volt.getRead()
    #     self.ui.Volt.setText(read+"V")
    #     print(read)
    #     # self.threadpool.start(volt.run())
    def selectMotor(self):
        motor = sendArduino('m1')
        self.threadpool.start(motor)
        self.read.run()
        sensor = self.read.getRead()
        volt = '0'
        current = '0'
        print(sensor)
        c = sensor.find('c')
        print("c = ",c)
        if c != -1:
            if sensor[0] == 'v':
                    volt = sensor[1:c]
            if sensor[c] =="c":
                    current = sensor[c+1:]
        print(volt)
        print(current)
        self.ui.Volt.setText(volt +"V")
        self.ui.Current.setText(current +"A")
    def selectESC(self):
        ESC = sendArduino('m2')
        self.threadpool.start(ESC)
        self.read.run()
        sensor = self.read.getRead()
        volt = '0'
        current = '0'
        print(sensor)
        c = sensor.find('c')
        print("c = ",c)
        if c != -1:
            if sensor[0] == 'v':
                    volt = sensor[1:c]
            if sensor[c] =="c":
                    current = sensor[c+1:]
        print(volt)
        print(current)
        self.ui.Volt.setText(volt +"V")
        self.ui.Current.setText(current +"A")
    def f_up (self):
        up = sendArduino('d3')
        self.threadpool.start(up)
        self.read.run()
        sensor = self.read.getRead()
        volt = '0'
        current = '0'
        print(sensor)
        c = sensor.find('c')
        print("c = ",c)
        if c != -1:
            if sensor[0] == 'v':
                    volt = sensor[1:c]
            if sensor[c] =="c":
                    current = sensor[c+1:]
        print(volt)
        print(current)
        self.ui.Volt.setText(volt +"V")
        self.ui.Current.setText(current +"A")
    def f_back (self):
        back = sendArduino('d4')
        self.threadpool.start(back)
        self.read.run()
        sensor = self.read.getRead()
        volt = '0'
        current = '0'
        print(sensor)
        c = sensor.find('c')
        print("c = ",c)
        if c != -1:
            if sensor[0] == 'v':
                    volt = sensor[1:c]
            if sensor[c] =="c":
                    current = sensor[c+1:]
        print(volt)
        print(current)
        self.ui.Volt.setText(volt +"V")
        self.ui.Current.setText(current +"A")    
    def f_right (self):
        right = sendArduino('d5')
        self.threadpool.start(right)
        self.read.run()
        sensor = self.read.getRead()
        volt = '0'
        current = '0'
        print(sensor)
        c = sensor.find('c')
        print("c = ",c)
        if c != -1:
            if sensor[0] == 'v':
                    volt = sensor[1:c]
            if sensor[c] =="c":
                    current = sensor[c+1:]
        print(volt)
        print(current)
        self.ui.Volt.setText(volt +"V")
        self.ui.Current.setText(current +"A")
    def f_left (self):
        left = sendArduino('d6')
        self.threadpool.start(left)
        self.read.run()
        sensor = self.read.getRead()
        volt = '0'
        current = '0'
        print(sensor)
        c = sensor.find('c')
        print("c = ",c)
        if c != -1:
            if sensor[0] == 'v':
                    volt = sensor[1:c]
            if sensor[c] =="c":
                    current = sensor[c+1:]
        print(volt)
        print(current)
        self.ui.Volt.setText(volt +"V")
        self.ui.Current.setText(current +"A")
    def f_stop (self):
        stop = sendArduino('d7')
        self.threadpool.start(stop)
        self.read.run()
        sensor = self.read.getRead()
        volt = '0'
        current = '0'
        print(sensor)
        c = sensor.find('c')
        print("c = ",c)
        if c != -1:
            if sensor[0] == 'v':
                    volt = sensor[1:c]
            if sensor[c] =="c":
                    current = sensor[c+1:]
        print(volt)
        print(current)
        self.ui.Volt.setText(volt +"V")
        self.ui.Current.setText(current +"A")
    def Speed1 (self):
        s_1 = sendArduino('s0')
        self.threadpool.start(s_1)
        self.read.run()
        sensor = self.read.getRead()
        volt = '0'
        current = '0'
        print(sensor)
        c = sensor.find('c')
        print("c = ",c)
        if c != -1:
            if sensor[0] == 'v':
                    volt = sensor[1:c]
            if sensor[c] =="c":
                    current = sensor[c+1:]
        print(volt)
        print(current)
        self.ui.Volt.setText(volt +"V")
        self.ui.Current.setText(current +"A")
    def Speed2 (self):
        s_2 = sendArduino('s1')
        self.threadpool.start(s_2)
        self.read.run()
        sensor = self.read.getRead()
        volt = '0'
        current = '0'
        print(sensor)
        c = sensor.find('c')
        print("c = ",c)
        if c != -1:
            if sensor[0] == 'v':
                    volt = sensor[1:c]
            if sensor[c] =="c":
                    current = sensor[c+1:]
        print(volt)
        print(current)
        self.ui.Volt.setText(volt +"V")
        self.ui.Current.setText(current +"A")
    def Speed3 (self):
        s_3 = sendArduino('s2')
        self.threadpool.start(s_3)
        self.read.run()
        sensor = self.read.getRead()
        volt = '0'
        current = '0'
        print(sensor)
        c = sensor.find('c')
        print("c = ",c)
        if c != -1:
            if sensor[0] == 'v':
                    volt = sensor[1:c]
            if sensor[c] =="c":
                    current = sensor[c+1:]
        print(volt)
        print(current)
        self.ui.Volt.setText(volt +"V")
        self.ui.Current.setText(current +"A")
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