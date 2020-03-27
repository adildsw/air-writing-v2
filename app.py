# -*- coding: utf-8 -*-
"""
Air-Writing v2

Created on Wed June 12 12:00:00 2019
Author: Adil Rahman | CVPR Unit - ISI Kolkata (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/adildsw/air-writing-v2

"""

import sys
import os
import webbrowser
import cv2
import numpy

from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QFrame, QWidget, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QDesktopWidget, QLabel, QPushButton

from camera import VideoStream
from pipeline_v2 import Pipeline
from calibration import CalibrationInterface


class MainGUI(QWidget):
    # ~~~~~~~~ constructor ~~~~~~~~
    def __init__(self):
        super().__init__()
        self.jsonConfig()
        self.init_pipeline()
        self.init_UI()
        
        return
    
    # ~~~~~~~~ initialize pipeline ~~~~~~~~
    def init_pipeline(self):
        self.pipeline = Pipeline()
        self.engine = 'EN'
        
        return
    
    # ~~~~~~~~ initialize ui ~~~~~~~~
    def init_UI(self):
        self.configColor = (200, 110, 0)
        self.configColor = self.configColor[::-1]
        
        self.result_main_label = 'Predicted Value(s): NA'
        self.result_alt1_label = 'Alternate Value(s) #1: NA'
        self.result_alt2_label = 'Alternate Value(s) #2: NA'
        
        # set properties
        self.setGeometry(0, 0, 0, 0)
        self.setStyleSheet('QWidget {background-color: #ffffff;}')
        self.setWindowIcon(QIcon('assets/logo.png'))
        self.setWindowTitle('Air-Writing')
        
        # create widgets
        # -- connect camera button --
        self.btn_conn = QPushButton('Connect Camera')
        self.btn_conn.setMinimumSize(350, 40)
        self.btn_conn_style_0 = 'QPushButton {background-color: #00a86c; border: none; color: #ffffff; font-family: ubuntu, arial; font-size: 16px;}'
        self.btn_conn_style_1 = 'QPushButton {background-color: #ff6464; border: none; color: #ffffff; font-family: ubuntu, arial; font-size: 16px;}'
        self.btn_conn.setStyleSheet(self.btn_conn_style_0)
        
        # -- connect file button --
        self.btn_file = QPushButton()
        self.btn_file.setMinimumSize(50, 40)
        self.btn_file_style_0 = 'QPushButton {background-color: #f1c40f; border: none; color: #ffffff; font-family: ubuntu, arial; font-size: 16px;}'
        self.btn_file_style_1 = 'QPushButton {background-color: #e67e22; border: none; color: #ffffff; font-family: ubuntu, arial; font-size: 16px;}'
        self.btn_file.setStyleSheet(self.btn_file_style_0)
        self.btn_file.setIcon(QIcon('assets/file.png'))
        self.btn_file.setIconSize(QSize(30, 30))
        
        # -- connect calibration button --
        self.btn_cal = QPushButton()
        self.btn_cal.setMinimumSize(50, 40)
        self.btn_cal_style_0 = 'QPushButton {background-color: rgb' + str(self.configColor) + '; border: none; color: #ffffff; font-family: ubuntu, arial; font-size: 16px;}'
        self.btn_cal_style_1 = 'QPushButton {background-color: #8c8c8c; border: none; color: #ffffff; font-family: ubuntu, arial; font-size: 16px;}'
        self.btn_cal.setStyleSheet(self.btn_cal_style_0)
        self.btn_cal.setIcon(QIcon('assets/config.png'))
        self.btn_cal.setIconSize(QSize(30, 30))
        
        # -- camera feed --
        self.cam_feed = QLabel()
        self.cam_feed.setMinimumSize(640, 480)
        self.cam_feed.setAlignment(Qt.AlignCenter)
        self.cam_feed.setFrameStyle(QFrame.StyledPanel)
        self.cam_feed.setStyleSheet('QLabel {background-color: #000000;}')
        
        # -- inference results --
        self.trace_label = QLabel('Traced Input')
        self.trace_label.setMinimumHeight(30)
        self.trace_label.setAlignment(Qt.AlignCenter)
        self.trace_label.setFrameStyle(QFrame.NoFrame)
        self.trace_label.setStyleSheet('QLabel {background-color: #ffffff; color: #646464; font-family: ubuntu, arial; font-size: 20px;}')
            
        self.trace_disp = QLabel('!')
        self.trace_disp.setMinimumHeight(260)
        self.trace_disp.setAlignment(Qt.AlignCenter)
        self.trace_disp.setFrameStyle(QFrame.StyledPanel)
        self.trace_disp.setStyleSheet('QLabel {background-color: #ffffff; color: #646464; font-family: ubuntu, arial; font-size: 200px;}')
            
        self.result_label = QLabel('Prediction Results')
        self.result_label.setMinimumHeight(30)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFrameStyle(QFrame.NoFrame)
        self.result_label.setStyleSheet('QLabel {background-color: #ffffff; color: #646464; font-family: ubuntu, arial; font-size: 20px;}')
            
        self.result_main = QLabel(self.result_main_label)
        self.result_main.setMinimumHeight(20)
        self.result_main.setAlignment(Qt.AlignLeft)
        self.result_main.setFrameStyle(QFrame.NoFrame)
        self.result_main.setStyleSheet('QLabel {background-color: #ffffff; color: #646464; font-family: ubuntu, arial; font-size: 16px; font-weight: bold}')
            
        self.result_alt1 = QLabel(self.result_alt1_label)
        self.result_alt1.setMinimumHeight(20)
        self.result_alt1.setAlignment(Qt.AlignLeft)
        self.result_alt1.setFrameStyle(QFrame.NoFrame)
        self.result_alt1.setStyleSheet('QLabel {background-color: #ffffff; color: #646464; font-family: ubuntu, arial; font-size: 16px;}')
        
        self.result_alt2 = QLabel(self.result_alt2_label)
        self.result_alt2.setMinimumHeight(20)
        self.result_alt2.setAlignment(Qt.AlignLeft)
        self.result_alt2.setFrameStyle(QFrame.NoFrame)
        self.result_alt2.setStyleSheet('QLabel {background-color: #ffffff; color: #646464; font-family: ubuntu, arial; font-size: 16px;}')
        
        # -- repository link button --
        self.btn_repo = QPushButton()
        self.btn_repo.setFixedSize(20, 20)
        self.btn_repo.setStyleSheet('QPushButton {background-color: none; border: none;}')
        self.btn_repo.setIcon(QIcon('assets/button_repo.png'))
        self.btn_repo.setIconSize(QSize(20, 20))
        self.btn_repo.setToolTip('Fork me on GitHub')
        
        # -- copyright --
        self.copyright = QLabel('\u00A9 2019 Indian Statistical Institute')
        self.copyright.setFixedHeight(20)
        self.copyright.setAlignment(Qt.AlignCenter)
        self.copyright.setStyleSheet('QLabel {background-color: #ffffff; font-family: ubuntu, arial; font-size: 14px;}')
        
        # -- indicator --
        self.indicator = QLabel()
        self.indicator.setFixedSize(20, 20)
        self.indicator.setAlignment(Qt.AlignCenter)
        self.indicator.setFrameStyle(QFrame.NoFrame)
        self.indicator_style_0 = 'QLabel {background-color: #646464;}'
        self.indicator_style_1 = 'QLabel {background-color: #8ce312;}'
        self.indicator_style_2 = 'QLabel {background-color: rgb(255, 200, 0);}'
        self.indicator.setStyleSheet(self.indicator_style_0)
        
        # font info
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.posMain = (15,400)
        self.posProgress = (300, 425)
        self.fontScale = 0.65
        self.fontColorMain = (255,255,255)
        self.fontColorProgress = (255,200,0)
        self.lineType = 2
        self.textMain = "Place your object on the marker and press 'a' to continue"
        self.textProgress = "1/5"
        self.progressCount = 1
        
        # create layouts
        h_box1 = QHBoxLayout()
        h_box1.addWidget(self.btn_conn)
#        h_box1.addWidget(self.btn_file)
        h_box1.addWidget(self.btn_cal)
        
        h_box2 = QHBoxLayout()
        h_box2.addWidget(self.trace_label)
        
        h_box3 = QHBoxLayout()
        h_box3.addWidget(self.trace_disp)
        
        h_box4 = QHBoxLayout()
        h_box4.addWidget(self.result_label)
        
        h_box5 = QHBoxLayout()
        h_box5.addWidget(self.result_main)
        
        h_box6 = QHBoxLayout()
        h_box6.addWidget(self.result_alt1)
        
        h_box7 = QHBoxLayout()
        h_box7.addWidget(self.result_alt2)
        
        h_box8 = QHBoxLayout()
        h_box8.addWidget(self.btn_repo)
        h_box8.addWidget(self.copyright)
        h_box8.addWidget(self.indicator)
        
        v_box1 = QVBoxLayout()
        v_box1.addLayout(h_box1)
        v_box1.addStretch()
        v_box1.addLayout(h_box2)
        v_box1.addLayout(h_box3)
        v_box1.addStretch()
        v_box1.addLayout(h_box4)
        v_box1.addLayout(h_box5)
        v_box1.addLayout(h_box6)
        v_box1.addLayout(h_box7)
        v_box1.addStretch()
        v_box1.addLayout(h_box8)
        
        v_box2 = QVBoxLayout()
        v_box2.addWidget(self.cam_feed)
        
        g_box0 = QGridLayout()
        g_box0.addLayout(v_box1, 0, 0, -1, 2)
        g_box0.addLayout(v_box2, 0, 2, -1, 4)
        
        self.setLayout(g_box0)
        
        # set slots for signals
        self.flg_conn = False
        self.flg_cal = False
        
        self.trace_img = None
        
        self.btn_conn.clicked.connect(self.connect) 
#        self.btn_file.clicked.connect(self.openFile)
        self.btn_cal.clicked.connect(self.calibrate)
        self.btn_repo.clicked.connect(self.openRepository)
        
        return
    
    def keyPressEvent(self, qKeyEvent):
        if qKeyEvent.key() == ord('A'):
            if self.calibrator:
                if self.calibrator._getCalIndex() < 5:
                    self.calibrator._increaseCalIndex();
                    self.progressCount = self.progressCount + 1
                    self.textProgress = str(self.progressCount) + '/5'
                if self.calibrator._getCalIndex() == 5:
                    self.calibrator._generateJSON()
                    self.pipeline._loadConfig()
                    self.configColor = self.calibrator._getCenterRGB()
                    self.configColor = self.configColor[::-1]
                    self.btn_cal_style_0 = 'QPushButton {background-color: rgb' + str(self.configColor) + '; border: none; color: #ffffff; font-family: ubuntu, arial; font-size: 16px;}'
                    self.progressCount = 1
                    self.textProgress = str(self.progressCount) + '/5'
                    self.calibrate()
                    
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setText("Calibration Successful")
                    msg.setWindowTitle("Success")
                    msg.exec_()
        else:
            super().keyPressEvent(qKeyEvent)
    
    def moveWindowToCenter(self):
        window_rect = self.frameGeometry()
        screen_cent = QDesktopWidget().availableGeometry().center()
        window_rect.moveCenter(screen_cent)
        self.move(window_rect.topLeft())
        
        return
    
    def connect(self):
        if not self.flg_cal:
            self.flg_conn = not self.flg_conn
            if self.flg_conn:
                self.btn_conn.setStyleSheet(self.btn_conn_style_1)
                self.btn_conn.setText('Disconnect Camera')
                self.indicator.setStyleSheet(self.indicator_style_1)
                self.video = VideoStream()
                self.timer = QTimer()
                self.timer.timeout.connect(self.update)
                self.timer.start(50)
            else:
                self.btn_conn.setStyleSheet(self.btn_conn_style_0)
                self.btn_conn.setText('Connect Camera')
                self.indicator.setStyleSheet(self.indicator_style_0)
                self.cam_feed.clear()
                self.timer.stop()
                self.video.clear()
        
        return
    
    def calibrate(self):
        if not self.flg_conn:
            self.flg_cal = not self.flg_cal
            if self.flg_cal:
                self.btn_cal.setStyleSheet(self.btn_cal_style_1)
                self.indicator.setStyleSheet(self.indicator_style_2)
                self.video = VideoStream()
                self.calibrator = CalibrationInterface()
                self.timer = QTimer()
                self.timer.timeout.connect(self.updateCalibration)
                self.timer.start(50)
            else:
                self.btn_cal.setStyleSheet(self.btn_cal_style_0)
                self.indicator.setStyleSheet(self.indicator_style_0)
                self.cam_feed.clear()
                self.timer.stop()
                self.video.clear()
                self.calibrator = None
        
        return
    
    def update(self):
        frame = self.video.getFrame(flip=1)
        if not frame is None:
            frame, trace, bi, fwd, rev = self.pipeline.run_inference(frame, self.engine, True)
            frame = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.cam_feed.setPixmap(QPixmap.fromImage(frame))
            if not trace is None and not bi == []:
                self.trace_img = trace
                self.result_main_label = 'Predicted Value(s): ' + str(bi)
                if not str(fwd) == str(bi):
                    self.result_alt1_label = 'Alternate Value(s) #1: ' + str(fwd)
                    if not str(rev) == str(bi) and not str(rev) == str(fwd):
                        self.result_alt2_label = 'Alternate Value(s) #2: ' + str(rev)
                    else:
                        self.result_alt2_label = 'Alternate Value(s) #2: NA'
                elif not str(rev) == str(bi):
                    self.result_alt1_label = 'Alternate Value(s) #1: ' + str(rev)
                    self.result_alt2_label = 'Alternate Value(s) #2: NA'
                else:
                    self.result_alt1_label = 'Alternate Value(s) #1: NA'
                    self.result_alt2_label = 'Alternate Value(s) #2: NA'
            
                toolTipString = 'Bi-directional Scan: ' + str(bi) + '\n' + 'Forward Scan: ' + str(fwd) + '\n' + 'Reverse Scan: ' + str(rev)
                self.result_label.setToolTip(toolTipString)
            else:
                self.result_label.setToolTip('')
                
        else:
            self.cam_feed.clear()
        
        if not self.trace_img is None:
            trace_img_disp = QImage(self.trace_img, self.trace_img.shape[1], self.trace_img.shape[0], self.trace_img.strides[0], QImage.Format_RGB888)
            self.trace_disp.setPixmap(QPixmap.fromImage(trace_img_disp))
            
            self.result_main.setText(self.result_main_label)
            self.result_alt1.setText(self.result_alt1_label)
            self.result_alt2.setText(self.result_alt2_label)
        
        return
    
    def updateCalibration(self):
        frame = self.video.getFrame(flip=1)
        if not frame is None:
            frame = self.calibrator._calibrate(frame)
            cv2.putText(frame, self.textMain, self.posMain, self.font, self.fontScale, self.fontColorMain, self.lineType)
            cv2.putText(frame, self.textProgress, self.posProgress, self.font, self.fontScale, self.fontColorProgress, self.lineType)
            frame = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.cam_feed.setPixmap(QPixmap.fromImage(frame))
        else:
            self.cam_feed.clear()
        
        return
    
    def jsonConfig(self):
        self.calibrator = CalibrationInterface()
        if not os.path.exists('config.json'):
            self.calibrator._generateDefaultJSON()
        self.configColor = self.calibrator._getCenterRGB()
        self.configColor = self.configColor[::-1]
        self.calibrator = None
            
        return
    
    def openFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Open Saved Character Co-ordinate File", "","Numpy Files (*.npy)", options=options)
        if fileName:
            pts = numpy.load(fileName)
            if not len(pts.shape) == 2:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Incorrect File Loaded")
                msg.setWindowTitle("Error")
                msg.exec_()
            elif not pts.shape[1] == 2:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Incorrect File Loaded")
                msg.setWindowTitle("Error")
                msg.exec_()
            else:
                self.readPts(pts)
        
        return
    
    def readPts(self, pts):
        trace, bi, fwd, rev = self.pipeline.run_inference_file(pts)
        if not trace is None and not bi == []:
            self.trace_img = trace
            self.result_main_label = 'Predicted Value(s): ' + str(bi)
            if not str(fwd) == str(bi):
                self.result_alt1_label = 'Alternate Value(s) #1: ' + str(fwd)
                if not str(rev) == str(bi) and not str(rev) == str(fwd):
                    self.result_alt2_label = 'Alternate Value(s) #2: ' + str(rev)
                else:
                    self.result_alt2_label = 'Alternate Value(s) #2: NA'
            elif not str(rev) == str(bi):
                self.result_alt1_label = 'Alternate Value(s) #1: ' + str(rev)
                self.result_alt2_label = 'Alternate Value(s) #2: NA'
            else:
                self.result_alt1_label = 'Alternate Value(s) #1: NA'
                self.result_alt2_label = 'Alternate Value(s) #2: NA'
        
            toolTipString = 'Bi-directional Scan: ' + str(bi) + '\n' + 'Forward Scan: ' + str(fwd) + '\n' + 'Reverse Scan: ' + str(rev)
            self.result_label.setToolTip(toolTipString)
        else:
            self.result_label.setToolTip('')
                
        
        if not self.trace_img is None:
            trace_img_disp = QImage(self.trace_img, self.trace_img.shape[1], self.trace_img.shape[0], self.trace_img.strides[0], QImage.Format_RGB888)
            self.trace_disp.setPixmap(QPixmap.fromImage(trace_img_disp))
            
            self.result_main.setText(self.result_main_label)
            self.result_alt1.setText(self.result_alt1_label)
            self.result_alt2.setText(self.result_alt2_label)
        
        return
    
    def openRepository(self):
        webbrowser.open('https://github.com/adildsw/air-writing-v2')
        
        return

    def closeEvent(self, event):
        if self.flg_conn or self.flg_cal:
            self.connect()
        sys.exit()
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    gui = MainGUI()
    gui.show()
    gui.setFixedSize(gui.size())
    gui.moveWindowToCenter()
    app.exec_()
