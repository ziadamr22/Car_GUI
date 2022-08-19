        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.showVolt)
        self.timer.start(1500)