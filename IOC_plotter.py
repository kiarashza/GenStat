#!/usr/bin/env python
# coding: utf-8

# In[260]:


#HETEROGENEOUS
# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.markers as markers

def plot(x, y_s):

# Data
    df=pd.DataFrame()

    df=pd.DataFrame({'x': np.array(x), 'hit_1' : y_s[:,0],
                'hit_5' : y_s[:,1],
                'hit_10' : y_s[:,2]
                    ,
                'hit_20' : y_s[:,3],
                 'hit_100' : y_s[:,4]

                    ,
                 'hit_500' : y_s[:,5]
                 })



    # multiple line plot
    #plt.plot( 'L', 'IMDB', data=df, marker='^', markerfacecolor='blue', markersize=10, color='blue', linewidth=2, linestyle='dashed', label='IMDB')
    plt.plot( 'x', 'hit_1', data=df, marker='8', markerfacecolor='darkgreen', markersize=7, color='darkgreen', linewidth=2,
             linestyle='dashed', label='hit_1')
    plt.plot( 'x', 'hit_5', data=df, marker='s', markerfacecolor='orange', markersize=7, color='orange', linewidth=2,
             linestyle='dashed', label='hit_5')
    plt.plot( 'x', 'hit_10', data=df, marker='^', markerfacecolor='firebrick', markersize=7, color='firebrick',
             linewidth=2, linestyle='dashed', label='hit_10')

    plt.plot( 'x', 'hit_20', data=df, marker='s', markerfacecolor='slategray', markersize=7, color='slategray', linewidth=2,
             linestyle='dashed', label='hit_20')
    plt.plot( 'x', 'hit_100', data=df, marker='^', markerfacecolor='olive', markersize=7, color='olive',
             linewidth=2, linestyle='dashed', label='hit_100')
    plt.plot( 'x', 'hit_500', data=df, marker='8', markerfacecolor='blue', markersize=7, color='blue',
             linewidth=2, linestyle='dashed', label='hit_500')

    #plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=2)
    #plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
    #plt.annotate('', xy=(8, 0.9025), xytext=(8, 0.88),arrowprops=dict(facecolor='black', shrink=0.02),)
    #plt.annotate('', xy=(4, 0.9317), xytext=(8, 0.88),arrowprops=dict(facecolor='black', shrink=0.02),


    plt.legend()
    # naming the x axis
    plt.xlabel('Intersection Number', fontsize=15)
    # naming the y axis
    plt.ylabel('Hit number', fontsize=15)
    # xi = [16, 32, 64, 128,256]
    # L = [16, 32, 64, 128,256]
    # plt.xticks(xi, L)
    plt.show()
    # giving a title to my graph
    # plt.title('Homogeneous Graphs')
    plt.savefig('IOC')


