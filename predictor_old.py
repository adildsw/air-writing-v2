"""
Created on Mon Jun 10 15:03:17 2019

@author: adildsw
"""

import os
import shutil
import numpy
import cv2
import re
import operator

from buffer_generator import generate, generate_segmented_data
from recognizer_v2 import Recognizer

if not 'recognizer' in globals():
    recognizer = Recognizer()

priority_matrix = []

priority_matrix_fwd = [
        [+0, -1, +1, +1, -1, -1, +1, -1, +1, +1], #0
        [-1, +0, -1, -1, +1, +1, +1, +1, -1, +1], #1
        [-1, -1, +0, +1, -1, -1, -1, -1, +1, -1], #2
        [-1, -1, -1, +0, -1, -1, -1, -1, -1, -1], #3
        [-1, -1, -1, -1, +0, -1, -1, -1, -1, +1], #4
        [-1, -1, -1, -1, -1, +0, -1, -1, +1, -1], #5
        [-1, -1, -1, -1, -1, -1, +0, -1, -1, -1], #6
        [+1, -1, +1, +1, -1, -1, -1, +0, +1, +1], #7
        [-1, -1, -1, -1, -1, -1, -1, -1, +0, -1], #8
        [-1, -1, +1, +1, -1, -1, -1, -1, +1, +0]  #9
    ]

priority_matrix_rev = [
        [+0, -1, -1, +1, -1, +1, +1, -1, -1, +1], #0
        [-1, +0, -1, -1, +1, -1, -1, +1, -1, +1], #1
        [-1, -1, +0, +1, -1, -1, -1, -1, +1, -1], #2
        [-1, -1, -1, +0, -1, -1, -1, -1, -1, -1], #3
        [-1, -1, -1, -1, +0, -1, -1, -1, -1, +1], #4
        [-1, -1, -1, -1, -1, +0, +1, -1, +1, -1], #5
        [-1, -1, -1, +1, -1, +1, +0, -1, +1, -1], #6
        [-1, -1, -1, -1, +1, -1, -1, +0, +1, +1], #7
        [-1, -1, -1, -1, -1, -1, -1, -1, +0, -1], #8
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, +0]  #9
    ]

def alphanumeric_sort(arr):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(arr, key = alphanum_key)

def prediction_pool(directory):
    ppool = []
    #Creating File List
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            files.append(filename)
    files = alphanumeric_sort(files)    
    files = files[11:] #Skipping smaller points
    
    #Predicting
    for idx, file in enumerate(files):
        if idx%4 == 0 or idx == (len(files) - 1):
            file_dir = os.path.join(directory, file)
            img = cv2.imread(file_dir)
            
            prob, pred, conf, ncp = recognizer.predict(img)
            if not pred == []:
                ppool.append([pred, file, conf, ncp])
    return ppool

def clean_ppool(ppool):
    cleaned_ppool = []
    count = 0
    start_idx = 0
    end_idx = -1
    
    for i in range(0, len(ppool) - 1):
        n1 = ppool[i][0]
        n2 = ppool[i + 1][0]
        
        if n1 == n2:
            if count == 0:
                start_idx = i
            count = count + 1
        
        if not n1 == n2:
            end_idx = i + 1
            count = 0
            temp_pool = []
            for j in range(start_idx, end_idx):
                temp_pool.append(ppool[j][2])
            idx, value = max(enumerate(temp_pool), key=operator.itemgetter(1))
            cleaned_ppool.append(ppool[start_idx + idx])
        
    return cleaned_ppool

def priority_filter(ppool):
    repeat = []
    count = 0
    #ppool = clean_ppool(ppool)
    print(ppool)
    if len(ppool) >= 2:
        for i in range(0, len(ppool) - 1):
            n1 = ppool[i][0]
            n2 = ppool[i + 1][0]
            if priority_matrix[n1][n2] > 0:
                if count >= 15: #For closed circuit character repitition
                    return repeat
                else:
                    repeat = []
                    count = 0
                    continue
            elif priority_matrix[n1][n2] < 0:
                return ppool[i]
            elif priority_matrix[n1][n2] == 0:
                if not repeat:
                    repeat = ppool[i]
                count = count + 1
                if count >= 30:
                    return repeat
        
        if len(ppool) > 0:        
            return ppool[len(ppool) - 1]
        else:
            return []
    elif len(ppool) == 1:
        return ppool[0]
    else:
        return []

def runPredictor(input_file):
    global priority_matrix
    buffer_size = 50
    points = numpy.load(input_file)
    rev_points = numpy.flip(points, axis=0)
    
    main_temp_dir = 'air_writing_temp/'
    fwd_temp_dir = main_temp_dir + 'front/'
    rev_temp_dir = main_temp_dir + 'rev/'
    
    
    if os.path.exists(main_temp_dir):
        shutil.rmtree(main_temp_dir)
    os.makedirs(main_temp_dir)
    
    full_result= []
    #~~~~~~~~~~~Forward Check~~~~~~~~~~~
    index = 0
    result = []
    priority_matrix = priority_matrix_fwd
    
    if os.path.exists(fwd_temp_dir):
        shutil.rmtree(fwd_temp_dir)
    os.makedirs(fwd_temp_dir)
    
    while index + 1 < len(points):
        temp_dir = fwd_temp_dir + str(index) + '/'
        os.makedirs(temp_dir)
        generate(points, temp_dir, index, buffer_size)
        ppool = prediction_pool(temp_dir)
        out = priority_filter(ppool)
        if out:
            shift = out[1][out[1].find('_') + 1 : out[1].find('.')]
            shift = int(shift)
            index = index + shift
            result.append(out)
            
            store_index = out[1][0 : out[1].find('_')]
            store_index = int(store_index)
            store_i = shift
            store_res = out[0]
            store_save_dir = 'generated_data/segmented/fwd/' + str(store_res) + '/'
            store_dump_name = input_file[input_file.find('/') + 1 : input_file.rfind('_')]
            generate_segmented_data(points, store_index, store_i, store_dump_name, store_save_dir)
        else:
            if index + 6 < len(points):
                index = index + 5
            else:
                index = len(points)
    full_result.append(result)
    print('Forward Check: ', result)
    
    #~~~~~~~~~~~Reverse Check~~~~~~~~~~~
    index = 0
    result = []
    priority_matrix = priority_matrix_rev
    
    if os.path.exists(rev_temp_dir):
        shutil.rmtree(rev_temp_dir)
    os.makedirs(rev_temp_dir)
    
    while index + 1 < len(rev_points):
        temp_dir = rev_temp_dir + str(index) + '/'
        os.makedirs(temp_dir)
        generate(rev_points, temp_dir, index, buffer_size)
        ppool = prediction_pool(temp_dir)
        out = priority_filter(ppool)
        if out:
            shift = out[1][out[1].find('_') + 1 : out[1].find('.')]
            shift = int(shift)
            index = index + shift
            result.append(out)
            
            store_index = out[1][0 : out[1].find('_')]
            store_index = int(store_index)
            store_i = shift
            store_res = out[0]
            store_save_dir = 'generated_data/segmented/rev/' + str(store_res) + '/'
            store_dump_name = input_file[input_file.find('/') + 1 : input_file.rfind('_')] + 'r'
            generate_segmented_data(rev_points, store_index, store_i, store_dump_name, store_save_dir)
        else:
            if index + 6 < len(rev_points):
                index = index + 5
            else:
                index = len(rev_points)
    
    result.reverse()
    full_result.append(result)
    print('Reverse Check: ', result)
    
    if os.path.exists(main_temp_dir):
        shutil.rmtree(main_temp_dir)
        
    #~~~~~~~~~~~Result Cleanup~~~~~~~~~~~
    fwd_res = []
    rev_res = []
    bi_res = []
    
    fwd_chopped = []
    rev_chopped = []
    
    for r in full_result[0]:
        fwd_res.append(r[0])
    for r in full_result[1]:
        rev_res.append(r[0])
    
    fwd_size = len(fwd_res)
    rev_size = len(rev_res)
    
    fwd_lim = fwd_size / 2
    rev_lim = rev_size / 2
    
    if fwd_size % 2 == 1:
        if not fwd_size == 1:
            fwd_lim = (fwd_size + 1) / 2
    if rev_size % 2 == 1:
        if not rev_size == 1:
            rev_lim = (rev_size - 1) / 2
    
    max_size = fwd_size
    while not (fwd_lim + rev_lim) >= max_size:
        if fwd_lim > rev_lim:
            rev_lim = rev_lim + 1
        else:
            fwd_lim = fwd_lim + 1
            
    fwd_chopped = fwd_res[:int(fwd_lim)]
    rev_chopped = rev_res[int(-rev_lim):]
    
    if len(fwd_chopped) > 0:
        for r in fwd_chopped:
            bi_res.append(r)
    if len(rev_chopped) > 0:
        for r in rev_chopped:
            bi_res.append(r)
    
    #Single Character Forward Run Ignore Bug Temporary Bypass
    if len(fwd_res) == 1:
        bi_res = fwd_res
        
    return bi_res, fwd_res, rev_res