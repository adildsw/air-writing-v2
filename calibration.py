#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:21:59 2019

@author: adildsw
"""

import json
import os
import cv2
import numpy

class CalibrationInterface(object):
    
    def __init__(self):
        self._calibration_index = 0
        self._thresh = 40
        self._bgr = [[0 for col in range(3)] for row in range(5)]
        self._minHSV = [[0 for col in range(3)] for row in range(5)]
        self._maxHSV = [[0 for col in range(3)] for row in range(5)]
        self._center_rgb = [0 for row in range(3)]
        self._pos = [(240, 320), (50, 60), (430, 60), (50, 580), (430, 580)] #positions for calibration
        
        return
    
    def _getHSV(self, frame):
        range_pixel = frame[self._pos[self._calibration_index]]
        hsv = cv2.cvtColor( numpy.uint8([[range_pixel]] ), cv2.COLOR_BGR2HSV)[0][0]
        minHSVTemp = numpy.array([hsv[0] - self._thresh, hsv[1] - self._thresh, hsv[2] - self._thresh])
        maxHSVTemp = numpy.array([hsv[0] + self._thresh, hsv[1] + self._thresh, hsv[2] + self._thresh])
        
        return minHSVTemp, maxHSVTemp
    
    def _getRGB(self, frame):
        range_pixel = frame[self._pos[self._calibration_index]]
        range_pixel = numpy.flip(range_pixel)
        
        return range_pixel
        
    
    def _calibrate(self, frame):
        height, width, channel = frame.shape
        
        frame_original = numpy.copy(frame)
        
        if self._calibration_index == 0:
            cv2.circle(frame, self._pos[self._calibration_index][::-1], 13, (255, 255, 255), thickness=3)
            cv2.drawMarker(frame, self._pos[self._calibration_index][::-1], (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)    
            self._minHSV[self._calibration_index], self._maxHSV[self._calibration_index] = self._getHSV(frame_original)
            self._center_rgb = self._getRGB(frame_original)
        elif self._calibration_index == 1:
            cv2.circle(frame, self._pos[self._calibration_index][::-1], 13, (255, 255, 255), thickness=3)
            cv2.drawMarker(frame, self._pos[self._calibration_index][::-1], (255, 200, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)
            self._minHSV[self._calibration_index], self._maxHSV[self._calibration_index] = self._getHSV(frame_original)
        elif self._calibration_index == 2:
            cv2.circle(frame, self._pos[self._calibration_index][::-1], 13, (255, 255, 255), thickness=3)
            cv2.drawMarker(frame, self._pos[self._calibration_index][::-1], (255, 200, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)
            self._minHSV[self._calibration_index], self._maxHSV[self._calibration_index] = self._getHSV(frame_original)
        elif self._calibration_index == 3:
            cv2.circle(frame, self._pos[self._calibration_index][::-1], 13, (255, 255, 255), thickness=3)
            cv2.drawMarker(frame, self._pos[self._calibration_index][::-1], (255, 200, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)
            self._minHSV[self._calibration_index], self._maxHSV[self._calibration_index] = self._getHSV(frame_original)
        elif self._calibration_index == 4:
            cv2.circle(frame, self._pos[self._calibration_index][::-1], 13, (255, 255, 255), thickness=3)
            cv2.drawMarker(frame, self._pos[self._calibration_index][::-1], (255, 200, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)
            self._minHSV[self._calibration_index], self._maxHSV[self._calibration_index] = self._getHSV(frame_original)
        
        return frame
    
    def _increaseCalIndex(self):
            self._calibration_index = self._calibration_index + 1
            
            return
    
    def _getCalIndex(self):
        
        return self._calibration_index
    
    def _getCenterRGB(self):
        centerRGB = (0, 0, 0)
        with open('config.json') as json_file:
            config = json.load(json_file)
            data = config['air-writing'][0]
            r = data['center_r']
            g = data['center_g']
            b = data['center_b']
            centerRGB = (r, g, b)
        return centerRGB
    
    def _generateDefaultJSON(self):
        config = {}
        config['air-writing'] = []
        
        config['air-writing'].append({
            'center_r': 200,
            'center_g': 110,
            'center_b': 0,
            
            'hsv_0_min_h': -25,
            'hsv_0_min_s': 215,
            'hsv_0_min_v': 187,
            'hsv_0_max_h': 55,
            'hsv_0_max_s': 295,
            'hsv_0_max_v': 267,
            
            'hsv_1_min_h': -23,
            'hsv_1_min_s': 197,
            'hsv_1_min_v': 211,
            'hsv_1_max_h': 57,
            'hsv_1_max_s': 277,
            'hsv_1_max_v': 291,
            
            'hsv_2_min_h': -26,
            'hsv_2_min_s': 215,
            'hsv_2_min_v': 144,
            'hsv_2_max_h': 54,
            'hsv_2_max_s': 295,
            'hsv_2_max_v': 224,
            
            'hsv_3_min_h': -26,
            'hsv_3_min_s': 215,
            'hsv_3_min_v': 180,
            'hsv_3_max_h': 54,
            'hsv_3_max_s': 295,
            'hsv_3_max_v': 260,
            
            'hsv_4_min_h': -26,
            'hsv_4_min_s': 215,
            'hsv_4_min_v': 154,
            'hsv_4_max_h': 54,
            'hsv_4_max_s': 295,
            'hsv_4_max_v': 234
        })
    
        with open('config.json', 'w') as outfile:
            json.dump(config, outfile)
        
        return
    
    def _generateJSON(self):
        if os.path.exists('config.json'):
            os.remove('config.json')
            
        config = {}
        config['air-writing'] = []
        
        config['air-writing'].append({
            'center_r': int(self._center_rgb[0]),
            'center_g': int(self._center_rgb[1]),
            'center_b': int(self._center_rgb[2]),
            
            'hsv_0_min_h': int(self._minHSV[0][0]),
            'hsv_0_min_s': int(self._minHSV[0][1]),
            'hsv_0_min_v': int(self._minHSV[0][2]),
            'hsv_0_max_h': int(self._maxHSV[0][0]),
            'hsv_0_max_s': int(self._maxHSV[0][1]),
            'hsv_0_max_v': int(self._maxHSV[0][2]),
            
            'hsv_1_min_h': int(self._minHSV[1][0]),
            'hsv_1_min_s': int(self._minHSV[1][1]),
            'hsv_1_min_v': int(self._minHSV[1][2]),
            'hsv_1_max_h': int(self._maxHSV[1][0]),
            'hsv_1_max_s': int(self._maxHSV[1][1]),
            'hsv_1_max_v': int(self._maxHSV[1][2]),
            
            'hsv_2_min_h': int(self._minHSV[2][0]),
            'hsv_2_min_s': int(self._minHSV[2][1]),
            'hsv_2_min_v': int(self._minHSV[2][2]),
            'hsv_2_max_h': int(self._maxHSV[2][0]),
            'hsv_2_max_s': int(self._maxHSV[2][1]),
            'hsv_2_max_v': int(self._maxHSV[2][2]),
            
            'hsv_3_min_h': int(self._minHSV[3][0]),
            'hsv_3_min_s': int(self._minHSV[3][1]),
            'hsv_3_min_v': int(self._minHSV[3][2]),
            'hsv_3_max_h': int(self._maxHSV[3][0]),
            'hsv_3_max_s': int(self._maxHSV[3][1]),
            'hsv_3_max_v': int(self._maxHSV[3][2]),
            
            'hsv_4_min_h': int(self._minHSV[4][0]),
            'hsv_4_min_s': int(self._minHSV[4][1]),
            'hsv_4_min_v': int(self._minHSV[4][2]),
            'hsv_4_max_h': int(self._maxHSV[4][0]),
            'hsv_4_max_s': int(self._maxHSV[4][1]),
            'hsv_4_max_v': int(self._maxHSV[4][2])
        })
    
        with open('config.json', 'w') as outfile:
            json.dump(config, outfile)
            
        return