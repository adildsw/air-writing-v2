#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:02:38 2019

@author: adildsw
"""

import os
import cv2
import numpy

class frameGenerator:
    
    def __init__(self):
        self.LW_FACTOR = 10 # LineWidth Factor = Max Dimension / LineWidth
        self.P_FACTOR = 4 # Padding Factor = Max Dimension / Padding
        
        return
    
    def generateFrame(self, points):
        assert (type(points) is numpy.ndarray and len(points) > 1 
                and points.shape[1] == 2)
        
        w, h = numpy.max(points, 0) - numpy.min(points, 0)
        max_dim = max(w, h)
        min_dim = min(w, h)
        
        line_width = int(round(max_dim/self.LW_FACTOR))
        padding = int(round(max_dim/self.P_FACTOR))
        
        normalized_points = numpy.copy(points)
        normalized_points -= (numpy.min(points, 0) - padding)
        
        frame = numpy.ones((int(h) + 2 * padding, int(w) + 2 * padding), 
                           dtype=numpy.uint8) * 255
        
        for i in range(len(points) - 1):
            cv2.line(frame, tuple(normalized_points[i]), 
                     tuple(normalized_points[i+1]), color=0, 
                     thickness=line_width, lineType=cv2.LINE_AA)
        
        # Squaring the frame
        w, h = frame.shape
        max_dim = max(w, h)
        min_dim = min(w, h)
        
        square_frame = numpy.copy(frame)
        
        dim_diff = max_dim - min_dim
        if dim_diff % 2 == 1:
            dim_diff += 1
        
        if w > h:
            extra_padding = numpy.ones((w, int(dim_diff/2)), 
                                       dtype=numpy.uint8) * 255
            square_frame = numpy.hstack((extra_padding, frame, extra_padding))
        elif h > w:
            extra_padding = numpy.ones((int(dim_diff/2), h), 
                                       dtype=numpy.uint8) * 255
            square_frame = numpy.vstack((extra_padding, frame, extra_padding))
            
        frame = square_frame
        
        return frame
    
    def generateAllFrames(self, points):
        frames = []
        
        for i in range(2, len(points)):
            points_subset = points[:i]
            frame = self.generateFrame(points_subset)
            frames.append(frame)
            
        return frames
    
    def generate(self, points, save_dir, index=0, buffer_size=0):
        if buffer_size == 0:
            buffer_size = len(points)
            
        points_subset = points[index : index + buffer_size]
        frames = self.generateAllFrames(points_subset)
        
        for idx, frame in enumerate(frames):
            filename = '{}_{}.jpg'.format(index, idx)
            filepath = os.path.join(save_dir, filename)
            
            cv2.imwrite(filepath, frame)
        
        return
    
    def generateFinal(self, directory, sub=False):
        if sub:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.npy'):
                        print(os.path.splitext(file)[0])
                        points = numpy.load(os.path.join(root, file))
                        img = fg.generateFrame(points)
                        cv2.imwrite(os.path.join(root, os.path.splitext(file)[0] + '.png'), img)
        else:
            for file in os.listdir(directory):
                if file.endswith('.npy'):
                    print(os.path.splitext(file)[0])
                    points = numpy.load(os.path.join(directory, file))
                    img = fg.generateFrame(points)
                    cv2.imwrite(os.path.join(directory, os.path.splitext(file)[0] + '.png'), img)
        
        return
    
if __name__ == "__main__":
    fg = frameGenerator()
    directory = "../../datasets/isi-air/isi-air-combined/"
    fg.generateFinal(directory, sub=True)
    