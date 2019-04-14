#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:12:58 2018

@author: sirapop
"""

class Normalizer:
    def __init__(self,X):
        self.data = X
        self.max_i = []
        self.min_i = []
        
    def reset_X(self,X):
        self.data = X
        self.max_i = []
        self.min_i = []
    
    def find_min_max(self):
        #find min and max for each column
        for k in range(0,len(self.data[0])):
            self.max_i.append(0.0)
            self.min_i.append(self.data[0][k])
        for i in range(0,len(self.data)):
            for j in range(0,len(self.data[i])):
                if self.data[i][j]>=self.max_i[j]:
                    self.max_i[j] = self.data[i][j]
                if self.data[i][j]<self.min_i[j]:
                    self.min_i[j] = self.data[i][j]
    
    def normalize(self):
        #linear scaling to unit range
        z = []
        self.find_min_max()
        for i in range(0,len(self.data)):
            new_x = []
            for j in range(0,len(self.data[i])):
                if self.max_i[j] == 0 and self.min_i[j] == 0:
                    x = 0
                else:
                    x = (self.data[i][j] - self.min_i[j])/(self.max_i[j]-self.min_i[j])
                new_x.append(x)
            z.append(new_x)
        return z