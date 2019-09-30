# -*- coding: utf-8 -*-
"""
Air-Writing v2.5 Pipeline.

Pipeline v1:
Created on Sat May 12 20:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/air-writing

Pipeline v2.5
Created on Mon Sep  2 18:00:00 2019
Author: Adil Rahman 

"""

from __future__ import division

import cv2
import numpy
import os
import re
import json

from predictor import predict_v3 as predict

class Pipeline(object):
    
    def __init__(self):
        self._minHSV = [[0 for col in range(3)] for row in range(5)]
        self._maxHSV = [[0 for col in range(3)] for row in range(5)]
        self._loadConfig()
        
        self._resizeHeight = 260
        self._resizeWidth = 347
        self._resizeDim = (self._resizeWidth, self._resizeHeight)
        
        self._x = -1
        self._y = -1
        self._dx = 0
        self._dy = 0
        self._vx = 0
        self._vy = 0
        self._histdx = []
        self._histdy = []
        self._points = []
        self._max_points = 400
        self._min_change = 10
        self._min_veloxy = 2.0
        self._marker_ctr = None
        self._marker_tip = None
        
        self._fps = 24
        
        self._render_marker = True
        self._render_trails = True
        
        self._opencv_version = int(cv2.__version__.split('.')[0])
        
        return
    
    def _loadConfig(self):
        with open('config.json') as json_file:
            config = json.load(json_file)
            data = config['air-writing'][0]
            for i in range(0, 5):
                self._minHSV[i][0] = data['hsv_' + str(i) + '_min_h']
                self._minHSV[i][1] = data['hsv_' + str(i) + '_min_s']
                self._minHSV[i][2] = data['hsv_' + str(i) + '_min_v']
                self._maxHSV[i][0] = data['hsv_' + str(i) + '_max_h']
                self._maxHSV[i][1] = data['hsv_' + str(i) + '_max_s']
                self._maxHSV[i][2] = data['hsv_' + str(i) + '_max_v']
        self._minHSV = numpy.array(self._minHSV)
        self._maxHSV = numpy.array(self._maxHSV)
        
        return
    
    def _marker_segmentation(self, frame):
        frame_blur = cv2.medianBlur(frame, 3)
        frame_HSV = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        
        maskHSV = cv2.inRange(frame_HSV, self._minHSV[0], self._maxHSV[0])
        for i in range(1, 5):
            maskHSV_new = cv2.inRange(frame_HSV, self._minHSV[i], self._maxHSV[i])
            maskHSV = maskHSV | maskHSV_new
        mask = cv2.medianBlur(maskHSV, 9)
        mask = cv2.dilate(mask, (9, 9))
        
        return mask
    
    def _marker_tip_identification(self, mask):
        if self._opencv_version == 2:
            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        else:
            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        
        if contours and len(contours) > 0:
            contour_max = sorted(contours, key = cv2.contourArea, reverse = True)[0]
            
            contour_roi = contour_max.reshape(contour_max.shape[0], contour_max.shape[2])
            contour_roi = sorted(contour_roi, key=lambda x:x[1])
            
            marker_tip = (contour_roi[0][0], contour_roi[0][1])
        else:
            contour_max = None
            marker_tip = None
        
        return [contour_max, marker_tip]
    
    def _trajectory_approximation(self, marker_tip, frame):
        image = None
        if marker_tip is None:
            self._x = -1
            self._y = -1
            self._dx = 0
            self._dy = 0
            self._vx = 0
            self._vy = 0
            self._histdx = []
            self._histdy = []
            self._points = []
        else:
            if len(self._histdx) >= self._fps:
                self._histdx.pop(0)
            if len(self._histdy) >= self._fps:
                self._histdy.pop(0)
            if len(self._points) > self._max_points:
                self._points.pop(0)
            
            if self._x < 0 or self._y < 0:
                self._x, self._y = marker_tip
            self._dx = abs(marker_tip[0] - self._x)
            self._dy = abs(marker_tip[1] - self._y)
            self._histdx.append(self._dx)
            self._histdy.append(self._dy)
            if self._dx > self._min_change or self._dy > self._min_change:
                self._points.append(marker_tip)
            self._x, self._y = marker_tip
            
            self._vx = numpy.floor(sum(self._histdx[-self._fps:]) / self._fps)
            self._vy = numpy.floor(sum(self._histdy[-self._fps:]) / self._fps)
            
            nodes = len(self._points)
            if nodes > 1:
                image = numpy.zeros((frame.shape[0], frame.shape[1]), dtype='uint8')
                for i in range(nodes-1):
                    cv2.line(image, self._points[i], self._points[i+1], (255, 255, 255), 4, cv2.LINE_AA)
        
        return image, self._points
    
    def _render(self, frame, ctr_draw=True, black=False):
        if not self._marker_ctr is None and ctr_draw:
            cv2.drawContours(frame, self._marker_ctr, -1, (0, 255, 0), 1)
        if not self._marker_tip is None and ctr_draw:
            cv2.circle(frame, self._marker_tip, 4, (255, 255, 0), -1)
        n = len(self._points)
        if n > 1:
            for i in range(n-1):
                if not black:
                    cv2.line(frame, self._points[i], self._points[i+1], (255, 255, 0), 4, cv2.LINE_AA)
                else:
                    cv2.line(frame, self._points[i], self._points[i+1], (0, 0, 0), 4, cv2.LINE_AA)
        
        return frame
    
    def run_inference(self, frame, engine='EN', mapping=True):
        bi = []
        fwd = []
        rev = []
        
        mask = self._marker_segmentation(frame)

        self._marker_ctr, self._marker_tip = self._marker_tip_identification(mask)

        image, pts = self._trajectory_approximation(self._marker_tip, frame)
        
        trace_img = None

        if not image is None and self._vx < self._min_veloxy and self._vy < self._min_veloxy:
            
            trace_img = 255 * numpy.ones(shape = frame.shape, dtype=numpy.uint8)
            trace_img = self._render(trace_img, ctr_draw = False, black = True)
            trace_img = cv2.resize(trace_img, self._resizeDim, interpolation = cv2.INTER_AREA)
            
            self._x = -1
            self._y = -1
            self._dx = 0
            self._dy = 0
            self._vx = 0
            self._vy = 0
            self._histdx = []
            self._histdy = []
            self._points = []
            self._marker_ctr = None
            self._marker_tip = None
            
            try:
                bi, fwd, rev = self.predict_data(pts)
            except:
                bi = []
                fwd = []
                rev = []

        frame = self._render(frame)
        
        return frame, trace_img, bi, fwd, rev
    
    def predict_data(self, points):
        if len(points) > 10:
            if not os.path.exists('generated_data/'):
                os.mkdir('generated_data/')
            if not os.path.exists('generated_data/segmented/'):
                os.mkdir('generated_data/segmented/')
                os.mkdir('generated_data/segmented/fwd/')
                os.mkdir('generated_data/segmented/rev/')
                for i in range(0, 10):
                    os.mkdir('generated_data/segmented/fwd/' + str(i) + '/')
                    os.mkdir('generated_data/segmented/rev/' + str(i) + '/')
                
            c = 0
            files = []
            for filename in os.listdir('generated_data/'):
                if filename.endswith('.npy'):
                    files.append(filename)
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            files = sorted(files, key = alphanum_key)
            if files:
                c = files[len(files) - 1]
                c = c[:c.find('_')]
                c = int(c)
                c = c + 1
            
            while os.path.isfile('generated_data/' + str(c) + '_' + str(len(points)) + '.npy'):
                c = c + 1
            numpy.save('generated_data/' + str(c) + '_' + str(len(points)) + '.npy', points)
        
            bi, fwd, rev = predict('generated_data/' + str(c) + '_' + str(len(points)) + '.npy')
            
            return bi, fwd, rev
