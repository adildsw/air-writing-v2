#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:47:18 2019

@author: adildsw
"""

import cv2
import numpy

def generate_buffer_v3(points, save_dir, index, padding=30, line_width=15):
    assert type(points) is numpy.ndarray and len(points) > 1 and points.shape[1] == 2
    
    for i in range(2, len(points)):
        point_subset = points[:i]
        point_subset = point_subset - (numpy.min(point_subset, 0) - padding)
        w, h = numpy.max(points, 0) - numpy.min(points, 0)
        
        image = numpy.ones((int(h) + 2 * padding, int(w) + 2 * padding), numpy.uint8) * 255
        
        for j in range(len(point_subset) - 1):
            cv2.line(image, tuple(point_subset[j]), tuple(point_subset[j+1]), 0, line_width, cv2.LINE_AA)
            
        cv2.imwrite(save_dir + '/{}_{}.jpg'.format(index, i), image)

def generate_segmented_data(points, index, i, dump_name, save_dir, padding=30, line_width=15):
    points = points[index : (index + i) + 1]
    point_subset = points
    point_subset = point_subset - (numpy.min(point_subset, 0) -padding)
    w, h = numpy.max(points, 0) - numpy.min(points, 0)
    
    image = numpy.ones((int(h) + 2 * padding, int(w) + 2 * padding), numpy.uint8) * 255
    for j in range(len(points) - 1):
        cv2.line(image, tuple(point_subset[j]), tuple(point_subset[j+1]), 0, line_width, cv2.LINE_AA)
    cv2.imwrite(save_dir + '/{}_{}_{}.jpg'.format(dump_name, index, i), image)
    
    numpy.save(save_dir + '/{}_{}_{}.npy'.format(dump_name, index, i), points)

def generate(points, save_dir, index=0, buffer_size=50):
    points = points[index:(index + buffer_size)]
    generate_buffer_v3(points, save_dir, index)