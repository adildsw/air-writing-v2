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

def prediction_pool(directory, model, frameskip):
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
        if idx%frameskip == 0 or idx == (len(files) - 1):
            file_dir = os.path.join(directory, file)
            img = cv2.imread(file_dir)
            
            prob, pred, conf, ncp = recognizer.predict(img, model)
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
#    print(ppool)
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

def runPredictor(input_file, model, frameskip):
    global priority_matrix
    buffer_size = 50
    points = numpy.load(input_file)
    rev_points = numpy.flip(points, axis=0)
    pts_len = len(points)
    half_mark = pts_len/2
    
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
        ppool = prediction_pool(temp_dir, model, frameskip)
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
#    print('Forward Check: ', result)
    
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
        ppool = prediction_pool(temp_dir, model, frameskip)
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
#    print('Reverse Check: ', result)
    
    if os.path.exists(main_temp_dir):
        shutil.rmtree(main_temp_dir)
        
    #~~~~~~~~~~~Result Cleanup~~~~~~~~~~~
    fwd_res = []
    rev_res = []
    bi_res = []
    
    fwd_split_mark = []
    rev_split_mark = []
    bi_split_mark = []
    
    for r in full_result[0]:
        fwd_res.append(r[0])
        split_mark = r[1]
        split_mark = split_mark.split('.')[0].split('_')
        split_start, inc = int(split_mark[0]), int(split_mark[1])
        split_end = split_start + inc
        fwd_split_mark.append('{}_{}'.format(split_start, split_end))
    for r in full_result[1]:
        rev_res.append(r[0])
        split_mark = r[1]
        split_mark = split_mark.split('.')[0].split('_')
        split_end, dec = pts_len - int(split_mark[0]), int(split_mark[1])
        split_start = split_end - dec
        rev_split_mark.append('{}_{}'.format(split_start, split_end))
    
#    print('Forward Split: {}\nReverse Split: {}'.format(fwd_split_mark, rev_split_mark))
    
    for i, r in enumerate(fwd_split_mark):
        start, end = int(r.split('_')[0]), int(r.split('_')[1])
        if (half_mark - start)/(end - start) > 0.5:
            bi_res.append(fwd_res[i])
            bi_split_mark.append(r)
        else:
            flag = True
            for s in rev_split_mark:
                start_rev, end_rev = int(s.split('_')[0]), int(s.split('_')[1])
                mid = (start + end)/2
                mid_rev = (start_rev + end_rev)/2
                if ((mid > start_rev and mid < end_rev) or
                       (mid_rev > start and mid_rev < end)):
                    flag = False
            if flag:
                bi_res.append(fwd_res[i])
                bi_split_mark.append(r)
    
    for i, r in enumerate(rev_split_mark):
        start, end = int(r.split('_')[0]), int(r.split('_')[1])
        if (end - half_mark)/(end - start) > 0.5:
            bi_res.append(rev_res[i])
            bi_split_mark.append(r)
        else:
            flag = True
            for s in fwd_split_mark:
                start_fwd, end_fwd = int(s.split('_')[0]), int(s.split('_')[1])
                mid = (start + end)/2
                mid_fwd = (start_fwd + end_fwd)/2
                if ((mid > start_fwd and mid < end_fwd) or
                       (mid_fwd > start and mid_fwd < end)):
                    flag = False
            if flag:
                bi_res.append(rev_res[i])
                bi_split_mark.append(r)
    
    for r in bi_split_mark:
        start, end = int(r.split('_')[0]), int(r.split('_')[1])
        for i, s in enumerate(bi_split_mark):
            if s != r:
                start_bi, end_bi = int(s.split('_')[0]), int(s.split('_')[1])
                mid = (start + end)/2
                mid_bi = (start_bi + end_bi)/2
                if ((mid > start_bi and mid < end_bi) or 
                    (mid_bi > start and mid_bi < end)):
                    bi_split_mark.remove(r)
                    bi_res.remove(bi_res[i])
    
    for i in range(0, len(bi_split_mark) - 1):
        for j in range(i + 1, len(bi_split_mark)):
            r = bi_split_mark[i]
            s = bi_split_mark[j]
            r_end = int(r.split('_')[1])
            s_end = int(s.split('_')[1])
            if s_end < r_end:
                bi_split_mark[i], bi_split_mark[j] = bi_split_mark[j], bi_split_mark[i]
                bi_res[i], bi_res[j] = bi_res[j], bi_res[i]
    
#    print(bi_res, fwd_res)
        
    return bi_res, fwd_res, rev_res