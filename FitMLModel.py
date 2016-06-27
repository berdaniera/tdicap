##################################3
from osgeo import gdal, gdalnumeric, ogr
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pa
import pickle
import re
import seaborn as sea
import simplejson
from sklearn.cross_validation import cross_val_score, KFold
from sklearn import neighbors
from sklearn import metrics
from sklearn.externals import joblib

# Bring in classified points
th = open("/home/aaron/Desktop/treehousedata.txt")
xo = simplejson.load(th)

# Bring in raster
fi = '/home/aaron/Desktop/2010448_1N1E08U4BAND/1n1e08u_4band.tif'
ras = gdal.Open(fi)
rasdat = ras.ReadAsArray()


# extract values
dat = pa.DataFrame(xo,columns=['type','x','y'])
dat.x = dat.x.astype(int)
dat.y = dat.y.astype(int)
sr = np.log(rasdat[3]*1./rasdat[0])
trees = sr[dat.x[dat.type=='T'],dat.y[dat.type=='T']]
houss = sr[dat.x[dat.type=='H'],dat.y[dat.type=='H']]

# plot histogram
bins = np.arange(-2,4,0.1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(houss,bins=bins,alpha=0.6,label="Houses")
ax.hist(trees,bins=bins,alpha=0.6,label="Trees")
ax.set_xlim([-0.5,2.5])
ax.legend(loc='upper right',fontsize=20)
ax.set_xlabel('Spectral index (NIR / Red)',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
#fig.show()
fig.savefig('test.pdf',format="pdf")

#sea.kdeplot(houss,shade=True,cut=0,bw=0.05)
#sea.kdeplot(trees,shade=True,cut=0,bw=0.05)
#sea.plt.show()

# KNN analysis
allval = sr[dat.x,dat.y]

nn = neighbors.KNeighborsClassifier(n_neighbors=10)
nn.fit(allval.reshape(-1,1),dat.type)
y_obs = dat.type
y_pred = nn.predict(allval.reshape(-1,1))

# save the model for loading later
joblib.dump(nn,'/home/aaron/Desktop/model/treemodel.pkl')
#load the model
# nn = joblib.load('/home/aaron/Desktop/model/treemodel.pkl')

scores = cross_val_score(nn,allval.reshape(-1,1),dat.type,cv=10,scoring='accuracy')
scores.mean()

metrics.confusion_matrix(y_obs,y_pred)

######################
# PREDICT WHOLE IMAGE
# Data for all points0
srr = sr.reshape(-1,1)
srr[np.isnan(srr)] = 0
srr[np.isinf(srr)] = 0
np.min(srr)
# k fold prediction

folds = np.array_split(srr,10)

classout = np.array([])
for f in range(len(folds)):
    classout = np.concatenate( (classout,nn.predict(folds[f])) )
    print (f+1.)/10

classout = classout.reshape(sr.shape)
classout.shape
#classout[:10,:10]

mask = np.zeros(classout.shape).astype(int)
mask[classout=='H'] = 255

# matrix to mask
realcol = np.array([rasdat[0,:,:],rasdat[1,:,:],rasdat[2,:,:]])
maskedcol = np.array([rasdat[0,:,:],rasdat[1,:,:],rasdat[2,:,:],mask])/255.

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(realcol.T[1000:2500,:3000,:])
#fig.show()
fig.savefig('/home/aaron/Desktop/MapOrig.pdf',format="pdf")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(maskedcol.T[1000:2500,:3000,:])
#fig.show()
fig.savefig('/home/aaron/Desktop/MapMask.pdf',format="pdf")
