#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:01:21 2018

@author: sirapop
"""

from __future__ import division
import datetime
import csv
from voter import Voter
from normalizer import Normalizer
import os
import sys

program_names = []
weights = []
class_list = []
X = [] #scores
y = [] #blank list
i = 0
vector_files = []
gram = -1 #set to 0 if positive. Set to -1 if negative.
for vf in range(1,len(sys.argv)):
    str_temp = sys.argv[vf]
    if str_temp.find('-gram') == -1:
        vector_files.append(str_temp)
    else:
        if str_temp.find('-gramneg') != -1:
            gram = -1
        else:
            gram = 0
use_voting = 0 #set to 1 if score voting will not be used.
cwd = os.getcwd()
f_setting = open(cwd + "/configuration.txt","rb")
settings = f_setting.readlines()
setting_flag = 0 #1 means right configuration found
for s in settings:
    s = s.rstrip()
    if s.find("<Gram")!= -1:
        if (s.find("->")!= -1 and gram < 0) or (s.find("+>")!=-1 and gram == 0):
            setting_flag = 1
            index_class_list = s.find("\t") + 1
            new_s = s[index_class_list:]
            class_list = new_s.split(',')
        else:
            setting_flag = 0
    elif s.find("Score_vote")!=-1:
        if "1" in s:
            use_voting = 0
        else:
            use_voting = 1
    else:
        if setting_flag > 0:
            index_n = s.find("\t")
            program_names.append(s[:index_n])
            weights.append(float(s[index_n+1:]))
program_names.append("PSO")
f_setting.close()

for vector_file in vector_files:
    with open(cwd + '/' + vector_file, 'rb') as csvfile:
        spamreader = csv.reader(csvfile,delimiter = ',')
        line = []
        for row in spamreader:
            line.append(row)
            if i > 0:
                j = 0
                sub_x = []
                while j < len(line[i]):
                    sub_x.append(float(line[i][j]))
                    j=j+1
                X.append(sub_x)
            i = i + 1
    csvfile.close()

    nm = Normalizer(X)
    X = nm.normalize()

    vote = Voter(X,y,gram)
    vote.set_class_list(class_list)
    nums = []
    for i in range(0,len(class_list)):
        nums.append(0)
    vote.set_each_class_nums(nums)

    #use pso
    vote.set_weights(weights)
    vote_results = vote.vote()
    other_results = vote.get_predicted()
    vote.set_use_voting(use_voting)
    vote.set_previous_votes(vote_results)
    acc = vote.weighted_vote()

    dt = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    
    ##write testing results
    predicted_tests = vote.get_predicted()
    vector_file = vector_file[:vector_file.find(".csv")]
    f_predicted = open("Predicted_" + vector_file + str(dt) + ".csv", "w+")
    for i in range(0,len(program_names)-1):
        f_predicted.write(str(program_names[i]) + ",")
    f_predicted.write("PSO\n")
    for i in range(0,len(other_results)):
        for j in range(0,len(other_results[i])):
            f_predicted.write(str(other_results[i][j]) + ",")
        f_predicted.write(predicted_tests[i] + "\n")
    f_predicted.close()
    
    print 'FINAL:\n'
    each_class = []
    each_class.append(0)
    for i in range(0,len(predicted_tests)):
        for j in range(0,len(class_list)):
            if len(each_class) < j+1:
                each_class.append(0)
            if predicted_tests[i] == class_list[j]:
                each_class[j] = each_class[j] + 1
                break
    for k in range(0,len(each_class)):
        print str(class_list[k]) + '\t:\t' + str(each_class[k]) + '\n'
    print '---------------------------------------------------------------\n'
print 'Thank you for using PSO-LocBact.\n'
