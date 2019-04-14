#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 13:01:01 2018

@author: biofilm
"""

from __future__ import division
import random
import math
from voter import Voter
import os
import csv
import datetime
from normalizer import Normalizer
import sys
#import sys

    
#-----------------------------------------------------------------------------------------------------------------------------
#This is a class for a particle in this method. In each particle, there exists a voter, which will be used for problem evaluation

class Particle:
    def __init__(self,X,y,votes,class_list,condition,gram):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.obj_best_i=0          # best objective function individual
        self.obj_i=0               # objective function individual
        self.r = random.random()    # rand()
        self.vote_i = votes
        self.gram = gram
        self.use_voting = condition #voting will be used or not
        self.num_dimensions = 0
        self.voter = Voter(X,y,gram)
        self.voter.set_previous_votes(self.vote_i)
        self.voter.set_class_list(class_list)
        self.voter.set_use_voting(self.use_voting)
        
    def set_weights(self,x0):
        self.voter.set_weights(x0)
        self.num_dimensions = len(x0)
        for i in range(0,self.num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])
    
    def get_voter(self):
        return self.voter
    
    # evaluate current fitness
    def evaluate(self):
        self.voter.set_weights(self.position_i)
        self.obj_i=self.voter.weighted_vote()

        # check to see if the current position is an individual best
        if self.obj_i > self.obj_best_i:
            self.pos_best_i=self.position_i
            self.obj_best_i=self.obj_i

    # update new particle velocity
    def update_velocity(self,pos_best_g,w,c1,c2):

        for i in range(0,self.num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self):
        for i in range(0,self.num_dimensions):
            self.position_i[i]=math.fabs(self.position_i[i]+self.velocity_i[i])

#-----------------------------------------------------------------------------------------------------------------------------

class PSO():
    def __init__(self,X,y,votes,num_particles,c1i,c1f,c2i,c2f,wi,wf,class_list, use_voting, gram):
        self.program_names = []
        self.condition = use_voting #if voting will be used
        self.each_predicted = [] #store final prediction of each tool
        self.each_class_counts = [] #number of samples in each class
        self.c1_i = c1i
        self.c1_f = c1f
        self.c2_i = c2i
        self.c2_f = c2f
        self.w1 = wi
        self.w2 = wf
        self.each_accs = []
        self.all_accs = []
        self.gram = gram
        if self.condition == 0:
            self.num_dimensions=1+(int(len(X[0])/len(class_list)))
        else:
            self.num_dimensions = int(len(X[0])/len(class_list))
        self.features = X
        self.labels = y
        self.obj_best_g=0                   # best objective function value for group
        self.pos_best_g=[]                   # best position for group
        self.index_best_g = 0
        self.class_list = class_list

        # establish the swarm
        self.swarm=[]
        for i in range(0,num_particles):
            x0 = []
            for j in range(0,self.num_dimensions):
                x0.append(random.random())
            p = Particle(X,y,votes,self.class_list,self.condition,self.gram)
            p.set_weights(x0)
            self.swarm.append(p)

    def set_each_class_count(self,counts):
        self.each_class_counts = counts

    def run(self, maxobj):
        # begin optimization loop
        i=0
        iteration = 0
        print 'Begin training a PSO-BactLoc weight vector...'
        while i < maxobj:
            #print 'Iteration = ' + str(iteration) + ' : \n'
            if i%10000 == 0:
                print 'Iteration = ' + str(iteration) + ' : \n'
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,len(self.swarm)):
                self.swarm[j].evaluate()
                i+=1

                # determine if current particle is the best (globally)
                if self.swarm[j].obj_i > self.obj_best_g:
                    self.pos_best_g=list(self.swarm[j].position_i)
                    self.obj_best_g=float(self.swarm[j].obj_i)
                    self.index_best_g = j
            
            #Update acceleration coefficients
            w = (self.w1-self.w2) * ((maxobj - i)/maxobj) + self.w2;
            c1 = (self.c1_f - self.c1_i) * (i/maxobj) + self.c1_i;
            c2 = (self.c2_f - self.c2_i) * (i/maxobj) + self.c2_i;

            # cycle through swarm and update velocities and position
            for j in range(0,len(self.swarm)):
                self.swarm[j].update_velocity(self.pos_best_g,w,c1,c2)
                self.swarm[j].update_position()
            
            iteration += 1
            if i%5000 == 0:
                print "Best of the group's objective function value: "
                print self.obj_best_g
                print '\nNumber of objective function calls: ' + str(i) + '\n'

        # print final results
        print 'FINAL:'
        best_votes = self.swarm[self.index_best_g].get_voter()
        print "\n----- Training accuracies -------\n"
        accs = best_votes.get_all_acc()
        voting = best_votes.get_voting_acc()
        accs.append(voting)
        self.all_accs = accs
        self.each_predicted = best_votes.get_predicted()
        for k in range(0,len(accs)):
            print self.program_names[k] + "\t: " + str(accs[k]) + "\n"
        print self.program_names[len(accs)] + "\t:" + str(self.obj_best_g)
        best_votes.set_each_class_nums(self.each_class_counts)
        self.each_accs = best_votes.get_each_class()
        for m in range(0,len(self.each_accs)):
            print "\n*-- Location " + str(self.class_list[m]) + "--*\n"
            for n in range(0,len(self.each_accs[m])):
                print self.program_names[n] + "\t: " + str(self.each_accs[m][n])
        print "\n--------------------------------\n"
        
    def get_weights(self):
        return self.pos_best_g
    
    def set_program_names(self, prog_names):
        self.program_names = prog_names
    
    def get_each_accs(self):
        return self.each_accs
    
    def get_all_accs(self):
        return self.all_accs
    
    def get_predicted(self):
        return self.each_predicted
    
    def get_acc(self):
        return self.obj_best_g
#-----------------------------------------------------------------------------------------------------------------------------

#--- RUN ----------------------------------------------------------------------+
program_names = []
X = [] #scores
y = [] #labels
i = 0
use_voting = 0 #set to 1 if score voting will not be used.
wi=0.9
wf=0.4
c1i=2.5
c1f=0.5
c2i=0.5
c2f=2.5
maxobj = 1000
maxiter = 0
class_list = []
vector_files = []
cwd = os.getcwd()
s_flag = 0
gram = 0 #-1 means negative, 0 means positive
for vf in range(1,len(sys.argv)):
    str_temp = sys.argv[vf]
    if str_temp.find('-gram') == -1:
        vector_files.append(str_temp)
    else:
        if str_temp.find('-gramneg') != -1:
            gram = -1
        else:
            gram = 0
f_setting = open(cwd + "/configuration.txt","rb")
settings = f_setting.readlines()

if gram == -1:
    gram_str = '<Gram->'
else:
    gram_str = '<Gram+>'
for s in settings:
    if s.find("w1")!= -1:
        w1 = float(s[s.find("w1")+3:])
    elif s.find("w2")!=-1:
        w2 = float(s[s.find("w2")+3:])
    elif s.find("c1i")!=-1:
        c1i = float(s[s.find("c1i")+4:])
    elif s.find("c1f")!=-1:
        c1f = float(s[s.find("c1f")+4:])
    elif s.find("c2i")!=-1:
        c2i = float(s[s.find("c2i")+4:])
    elif s.find("c2f")!=-1:
        c2f = float(s[s.find("c2f")+4:])
    elif s.find("particle num")!=-1:
        particle_num = int(s[s.find("particle num")+len("particle num"):])
    elif s.find("MAXOBJ")!=-1:
        maxobj = int(s[s.find("MAXOBJ")+len("MAXOBJ"):])
    elif s.find("MAXITER")!=-1:
        if s.find("-")!=-1:
            maxiter = 0
        else:
            maxiter = int(s[s.find("MAXOBJ")+len("MAXOBJ"):])
    elif s.find("Score_vote")!=-1:
        if "1" in s:
            use_voting = 0
        else:
            use_voting = 1
    elif s.find(gram_str)!=-1:
        s_flag = 1
        prog_list = s[s.find('\t')+1:].rstrip()
        class_list = prog_list.split(',')
    elif s.find('<Gram')!=-1 and s_flag > 0:
        s_flag = 0
    else:
        if s_flag > 0:
            index_n = s.find("\t")
            if s.find('#') == -1:
                program_names.append(s[:index_n])
program_names.append("PSO")
f_setting.close()

for vector_file in vector_files:
    with open(cwd + '/' + vector_file, 'rb') as csvfile:
        spamreader = csv.reader(csvfile,delimiter = ',')
        line = []
        for row in spamreader:
            line.append(row)
            if i > 0:
                y.append(line[i][0])
                j = 1
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
    vote_results = vote.vote()

    #use pso
    swarms = PSO(X,y,vote_results,25,c1i,c1f,c2i,c2f,wi,wf,class_list,use_voting,gram)
    swarms.set_program_names(program_names)
    swarms.set_each_class_count(vote.get_each_class_nums())
    swarms.run(maxobj)
    acc = swarms.get_acc()
    predicted_trains = swarms.get_predicted()
    each_accs = swarms.get_each_accs()
    weights = swarms.get_weights()
    
    #write to summary files
    dt = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    all_accs_train = swarms.get_all_accs()
    summary = "summary_" + str(dt) + ".txt"
    f_summary = open(summary, "w+")
    f_summary.write("Number of allowable objective function calls\t:\t" + str(maxobj) + "\n")
    if use_voting == 1:
        f_summary.write("Score voting in the mix\t:\tNo.\n")
    else:
        f_summary.write("Score voting in the mix\t:\tYes.\n")
    f_summary.write("-- Training -- (" + str(dt) + ")\n")
    f_summary.write("Program\t\t\tWeight\t\t\tOverall accuracy\n")
    for i in range(0,len(weights)):
        f_summary.write(program_names[i] + "\t\t" + str(weights[i]) + "\t\t" + str(all_accs_train[i]) + "\n")
    for m in range(0,len(each_accs)):
        f_summary.write("\n*-- Location " + str(class_list[m]) + "--*\n")
        for n in range(0,len(each_accs[m])):
            f_summary.write(program_names[n] + "\t: " + str(each_accs[m][n]) + "\n")
        f_summary.write("\n--------------------------------\n")
    f_summary.write("#### PSO accuracy :\t" + str(acc) + " ####\n")
    f_summary.write("\n")
    f_summary.close()

    #write the predicted answers
    other_results = vote.get_predicted()
    f_predicted = open("Predicted_train_" + str(dt) + ".csv", "w+")
    f_predicted.write("Label,")
    for i in range(0,len(program_names)-1):
        f_predicted.write(str(program_names[i]) + ",")
    f_predicted.write("PSO\n")
    for i in range(0,len(predicted_trains)):
        f_predicted.write(y[i] + ",")
        for j in range(0,len(other_results[i])):
            f_predicted.write(str(other_results[i][j]) + ",")
        f_predicted.write(predicted_trains[i] + "\n")
    f_predicted.close()