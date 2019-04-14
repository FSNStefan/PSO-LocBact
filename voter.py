#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 19:09:30 2018

@author: sirapop
"""

from __future__ import division

#-----------------------------------------------------------------------------------------------------------------------------
#This Voter class is used for score voting and weighted voting. The weight vector in a particle will be used here. 

class Voter:
    def __init__(self,X,y,gram):
        self.features = X
        self.gram = gram
        self.labels = y
        self.each_predicted = []
        self.weights = []
        self.answers = []
        self.votes = []
        self.use_voting = 0
        self.voting_acc = 0
        self.class_list = []
        self.all_accs = []
        self.each_class_accs = []
        self.each_class_nums = []
    
    def set_class_list(self,class_list):
        self.class_list = class_list
    
    def get_each_class(self):
        #calculate accuracy for classification of each classifier for each class
        for i in range(0,len(self.each_class_accs)):
            for j in range(0,len(self.each_class_accs[i])):
                if self.each_class_nums[i] > 0:
                    self.each_class_accs[i][j] = float(self.each_class_accs[i][j])/float(self.each_class_nums[i])
        return self.each_class_accs
    
    def set_each_class_nums(self,nums):
        #set the number of samples in each class
        self.each_class_nums = nums
        
    def check_each_accs(self,predicted,labels):
        accs = []
        for j in range(0,len(self.class_list)):
            accs.append(0)
        for i in range(0,len(predicted)):
            if predicted[i] == labels[i]:
                for j in range(0,len(self.class_list)):
                    if labels[i] == self.class_list[j]:
                        accs[j] += 1
                        break
        for j in range(0,len(self.each_class_nums)):
            accs[j] = accs[j]/self.each_class_nums[j]
        return accs
        
    def get_each_class_nums(self):
        return self.each_class_nums
    
    def set_weights(self,x0):
        #set weights calculated by PSO
        self.weights = x0
        self.each_class_accs = []
        for i in range(0,len(self.class_list)):
            self.each_class_nums.append(0)
            each_class = []
            for j in range(0,len(x0)):
                each_class.append(0)
            self.each_class_accs.append(each_class)
        
    def set_use_voting(self,condition):
        self.use_voting = condition
    
    def reset(self,X,y,x0):
        self.features = X
        self.labels = y
        self.weights = x0
        self.answers = []
        
    def set_previous_votes(self,previous_votes):
        self.votes = previous_votes
    
    def find_max(self,arr):
        #return the index of location that contains the maximum score
        max_mem = 0
        max_index = 0
        for i in range(0,len(arr)):
            if arr[i] > max_mem:
                max_mem = arr[i]
                max_index = i
        return max_index
    
    def vote(self):
        #create a vector of five scores generated from summation of scores from all tools for each location 
        new_X = []
        count_limit = 3
        if self.gram == -1:
            count_limit = 4
        for i in range(0,len(self.features)):
            counter = 0
            k = 0
            row = []
            each_tool = []
            sub_x = []
            #this function also counts number of samples in each location
            for l in range(0,len(self.class_list)):
                sub_x.append(0.0)
                if len(self.labels) > 0:
                    if self.labels[i] == self.class_list[l]:
                        self.each_class_nums[l] = self.each_class_nums[l] + 1
            for j in range(0,len(self.features[i])):
                if counter > count_limit:
                    max_index = self.find_max(each_tool)
                    row.append(self.class_list[max_index])
                    each_tool = []
                    counter = 0
                    k = k+1
                sub_x[counter] = self.features[i][j] + sub_x[counter]
                each_tool.append(self.features[i][j])
                counter = counter + 1
            max_index = self.find_max(each_tool)
            row.append(self.class_list[max_index])
            each_tool = []
            max_index = self.find_max(sub_x)
            row.append(self.class_list[max_index])
            self.answers.append(row)
            new_X.append(sub_x)
        return new_X
        
    def get_all_acc(self):
        #calculate accuracy for each tool
        for i in range(0,len(self.all_accs)):
            self.all_accs[i] = float(self.all_accs[i])/float(len(self.features))
        return self.all_accs
        
    def get_voting_acc(self):
        acc = 0
        for i in range(0,len(self.votes)):
            max_index = self.find_max(self.votes[i])
            if self.class_list[max_index] == self.labels[i]:
                acc = acc + 1
                temp_t = len(self.each_class_accs[max_index])-1
                self.each_class_accs[max_index][temp_t] = self.each_class_accs[max_index][temp_t] + 1
        acc = float(acc)/len(self.votes)
        return acc
    
    def get_predicted(self):
        return self.answers

    def weighted_vote(self):
        new_X = []
        self.all_accs = []
        self.answers = []
        acc = 0
        k = 0
        count_limit = 3
        if self.gram == -1:
            count_limit = 4
        for n in range(0,len(self.features[0])):
            if k > count_limit:
                self.all_accs.append(0.0)
                k = 0
            k = k + 1
        self.all_accs.append(0.0)
        for i in range(0,len(self.features)):
            counter = 0
            k = 0
            sub_x = []
            tool = []
            for l in range(0,len(self.class_list)):
                sub_x.append(0.0)
            for j in range(0,len(self.features[i])):
                if counter>count_limit: #run through all locations
                    counter = 0
                    temp_index = self.find_max(tool)
                    if len(self.labels) > 0:
                        if self.class_list[temp_index] == self.labels[i]:
                            self.all_accs[k] = self.all_accs[k] + 1
                            self.each_class_accs[temp_index][k] = self.each_class_accs[temp_index][k] + 1
                    tool = []
                    k = k +1
                x = self.features[i][j] * self.weights[k]
                tool.append(self.features[i][j])
                sub_x[counter] = sub_x[counter] + x
                counter = counter+1
            counter = 0
            temp_index = self.find_max(tool)
            if len(self.labels) > 0:
                if self.class_list[temp_index] == self.labels[i]: #last program included
                    self.all_accs[k] = self.all_accs[k] + 1
                    self.each_class_accs[temp_index][k] = self.each_class_accs[temp_index][k] + 1
            if self.use_voting == 0: #score voting
                for m in range(0,len(self.votes[i])):
                    x = self.votes[i][m] * self.weights[k+1]
                    sub_x[counter] = sub_x[counter] + x
                    counter = counter+1
            new_X.append(sub_x) #new calculated score vector
            max_index = self.find_max(sub_x)
            if len(self.labels) > 0:
                if self.class_list[max_index] == self.labels[i]:
                    acc = acc + 1
            self.answers.append(self.class_list[max_index])
        acc = float(acc)/float(len(self.features))
        return acc