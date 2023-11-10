#!/usr/bin/env python
# coding: utf-8

# In[260]:


#HETEROGENEOUS
# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.markers as markers

#=====================================================================================================================
# Data VGAE*
df=pd.DataFrame({'d': np.array([32,64,128,256, 512]), 'DBLP.1' : np.array([ 90.4980, 90.0244, 90.8589, 89.8561, 87.2226]),#np.array([0.893649, 0.904980, 0.900244, 0.908589, 0.898561, 0.872226]),
                'ACM.1' : np.array([ 92.9484, 92.4414, 91.5098, 90.6775, 88.8305]), #np.array([0.93423, 0.928466, 0.926216, 0.915436,  0.917247, 0.898315])
                'Cora.1': np.array([92.8466, 92.6216, 91.5436,  91.7247, 89.8315])# np.array([0.93423, 0.928466, 0.926216, 0.915436,  0.917247, 0.898315])
                    ,
                 'Citeseer.1': np.array([90.9593, 91.2262, 92.1372, 91.3324, 90.1224])#np.array([0.896491, 0.909593, 0.912262, 0.921372, 0.913324, 0.901224])

                    ,
                 'Pubmed.1': np.array([ 95.8199, 96.0370, 96.0900,96.1494, 95.5782])#np.array([0.959012, 0.958199, 0.960370, 0.960900,0.961494, 0.955782])

                    ,
                 'IMDB.1': np.array([ 84.5310, 85.5170, 85.3500, 84.5137, 85.5789]) #np.array([0.827125, 0.845310, 0.855170, 0.853500, 0.845137, 0.855789])
                 })

# DGLFRM
# df=pd.DataFrame({'d': np.array([32,64,128,256, 512]), 'ACM' : np.array([ 95.1947,95.9175 , 97.0866, 96.3705, 96.15]),#np.array([0.893649, 0.904980, 0.900244, 0.908589, 0.898561, 0.872226]),
#                 'DBLP' : np.array([ 94.77, 95.92, 95.84,96.02, 95.62, ]), #np.array([0.93423, 0.928466, 0.926216, 0.915436,  0.917247, 0.898315])
#                 'Cora': np.array([91.20, 93.51, 93.39,  93.71, 93.40])# np.array([0.93423, 0.928466, 0.926216, 0.915436,  0.917247, 0.898315])
#                     ,
#                  'Citeseer': np.array([88.56, 90.03, 90.94, 92.58, 91.75])#np.array([0.896491, 0.909593, 0.912262, 0.921372, 0.913324, 0.901224])
#
#                     ,
#                  'Pubmed': np.array([ 96.68, 96.93, 97.25,97.25,97.48])#np.array([0.959012, 0.958199, 0.960370, 0.960900,0.961494, 0.955782])
#
#                     ,
#                  'IMDB': np.array([ 83.42,85.20,85.92 ,87.24, 87.78 ])
#                  })
# df=pd.DataFrame({'d': [1,2,3,4], 'IMDB' : [0.85561,0.87,0.8852,0.8882],
#                 'DBLP' : [0.91,0.9170,0.9200,0.9317],
#                 'ACM': [0.9400,0.9694,0.9600,0.9579] })
#


plt.ylim(84, 97)
# multiple line plot
#plt.plot( 'L', 'IMDB', data=df, marker='^', markerfacecolor='blue', markersize=10, color='blue', linewidth=2, linestyle='dashed', label='IMDB')
plt.plot( 'd', 'ACM.1', data=df, marker='8', markerfacecolor='darkgreen', markersize=7, color='darkgreen', linewidth=2,
         linestyle='dashed', label='ACM.1')
plt.plot( 'd', 'DBLP.1', data=df, marker='s', markerfacecolor='orange', markersize=7, color='orange', linewidth=2,
         linestyle='dashed', label='DBLP.1')
plt.plot( 'd', 'IMDB.1', data=df, marker='^', markerfacecolor='firebrick', markersize=7, color='firebrick',
         linewidth=2, linestyle='dashed', label='IMDB.1')

plt.plot( 'd', 'Pubmed.1', data=df, marker='p', markerfacecolor='slategray', markersize=7, color='slategray', linewidth=2,
         linestyle='dashed', label='Pubmed.1')
plt.plot( 'd', 'Cora.1', data=df, marker='H', markerfacecolor='olive', markersize=7, color='olive',
         linewidth=2, linestyle='dashed', label='Cora.1')
plt.plot( 'd', 'Citeseer.1', data=df, marker='X', markerfacecolor='blue', markersize=7, color='blue',
         linewidth=2, linestyle='dashed', label='Citeseer.1')

#plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=2)
#plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
#plt.annotate('', xy=(8, 0.9025), xytext=(8, 0.88),arrowprops=dict(facecolor='black', shrink=0.02),)
#plt.annotate('', xy=(4, 0.9317), xytext=(8, 0.88),arrowprops=dict(facecolor='black', shrink=0.02),


plt.legend()
# naming the x axis
plt.xlabel('Dimension of the Node Latent Representation(d\')', fontsize=15)
# naming the y axis
plt.ylabel('Area Under Curve (AUC)', fontsize=15)
xi = [ 32, 64, 128,256, 512]
L = [32, 64, 128,256,512]
plt.xticks(xi, L)
# plt.show()
# giving a title to my graph
# plt.title('Homogeneous Graphs')
plt.savefig('realted_work.1.png')

# In[247]:

# HETEROGENEOUS
# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.markers as markers
plt.clf()

#---------------------------------------------------------------------------------
# VGAE* Data
df = pd.DataFrame({'L': [1, 2, 4,  6,  8,  10],
                   'IMDB': [85.52, 88.08, 88.49, 89.48, 89.92, 89.72],
                   'DBLP': [90.02, 92.01, 92.80, 93.07, 92.53, 93.03],
                   'ACM': [92.44, 93.43 , 92.95, 93.77, 94.08, 93.95]})
# #  DLFRM Data
# df = pd.DataFrame({'L': [1, 2, 4,  6,  8,  10],
#                    'IMDB': [85.20, 87.48, 87.81, 89.15, 89.98,89.27 ],
#                    'DBLP': [96.01, 96.37, 96.59, 96.76, 96.73, 96.77],
#                    'ACM': [95.49,  97.38, 98.04, 98.10, 98.33, 98.48]})
#---------------------------------------------------------------------------------
plt.ylim(85, 97)
# multiple line plot
# plt.plot( 'L', 'IMDB', data=df, marker='^', markerfacecolor='blue', markersize=10, color='blue', linewidth=2, linestyle='dashed', label='IMDB')
plt.plot('L', 'ACM', data=df, marker='8', markerfacecolor='darkgreen', markersize=7, color='darkgreen', linewidth=2,
         linestyle='dashed', label='ACM.1')
plt.plot('L', 'DBLP', data=df, marker='s', markerfacecolor='orange', markersize=7, color='orange', linewidth=2,
         linestyle='dashed', label='DBLP.1')
plt.plot('L', 'IMDB', data=df, marker='^', markerfacecolor='firebrick', markersize=7, color='firebrick',
         linewidth=2, linestyle='dashed', label='IMDB.1')

# plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=2)
# plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
# plt.annotate('', xy=(8, 0.9025), xytext=(8, 0.88),arrowprops=dict(facecolor='black', shrink=0.02),)
# plt.annotate('', xy=(4, 0.9317), xytext=(8, 0.88),arrowprops=dict(facecolor='black', shrink=0.02),


plt.legend()
# naming the x axis
plt.xlabel('# of Latent Layers (L)', fontsize=15)
# naming the y axis
plt.ylabel('Area Under Curve (AUC)', fontsize=15)
xi = list(range(11))
L = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.xticks(xi, L)

# giving a title to my graph
plt.title('Homogenized Graphs')
plt.savefig('Hetero_L.1.png')

# In[247]:

plt.clf()
# Homogeneous
# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.markers as markers
import matplotlib.colors as mcolors

# Data VGAE*
df = pd.DataFrame({'L': [1, 2,  4, 6,  8, 10],
                   'Pubmed': [96.03, 96.50,  96.65, 96.67,96.59, 96.57],
                   'Cora': [92.6216,93.44, 94.3265,94.2642,94.2642, 93.9470],
                   'Citeseer': [91.2262, 91.9652, 92.4801, 92.4154 ,93.038, 93.056 ]})

# # Data DGLFRM
# df = pd.DataFrame({'L': [1, 2,  4, 6,  8, 10],
#                    'Pubmed': [96.93, 97.56,  97.75, 98.32, 97.59, 97.48 ],
#                    'Citeseer': [90.09,91.52, 91.64,92.39,91.28, 91.18],
#                    'Cora': [93.54, 93.57, 94.67, 94.92, 94.19, 93.1, ]})

plt.ylim(88, 100)
# multiple line plot
# plt.plot( 'L', 'IMDB', data=df, marker='^', markerfacecolor='blue', markersize=10, color='blue', linewidth=2, linestyle='dashed', label='IMDB')
plt.plot('L', 'Pubmed', data=df, marker='s', markerfacecolor='slategray', markersize=7, color='slategray', linewidth=2,
         linestyle='dashed', label='Pubmed.1')
plt.plot('L', 'Cora', data=df, marker='^', markerfacecolor='olive', markersize=7, color='olive',
         linewidth=2, linestyle='dashed', label='Cora.1')
plt.plot('L', 'Citeseer', data=df, marker='8', markerfacecolor='blue', markersize=7, color='blue',
         linewidth=2, linestyle='dashed', label='Citeseer.1')

# plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=2)
# plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
# plt.annotate('', xy=(8, 0.9025), xytext=(8, 0.88),arrowprops=dict(facecolor='black', shrink=0.02),)
# plt.annotate('', xy=(4, 0.9317), xytext=(8, 0.88),arrowprops=dict(facecolor='black', shrink=0.02),


plt.legend()
# naming the x axis
plt.xlabel('# of Latent Layers (L)', fontsize=15)
# naming the y axis
plt.ylabel('Area Under Curve (AUC)', fontsize=15)
xi = list(range(11))
L = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.xticks(xi, L)

# giving a title to my graph
plt.title('Homogeneous Graphs')
plt.savefig('Homo_L.1.png')
