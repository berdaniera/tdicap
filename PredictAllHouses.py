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

# Bring in model
nn = joblib.load('modelFit/treemodel.pkl')

# Bring in all house points from each image
roo = '/home/vagrant/capstone/outdat/'
di = os.listdir(roo)
dirs = [d for d in di if '_housedat.pkl' in d]

dat = []
cnt = 0
ctot = 0
for d in dirs:
    cnt += 1
    tmp = pickle.load(open(roo+d, "rb"))
    coos = [t['corner'] for t in tmp]
    vals = [[len(x[x=="T"]),len(x)] for x in [nn.predict(np.array(t['sr']).reshape(-1,1)) for t in tmp]]
    dat.append([c+v for c, v in zip(coos,vals)])
    ctot += len(coos)
    print cnt, len(coos), ctot

datx = [item for sublist in dat for item in sublist]

datout = pa.DataFrame(datx,columns=['x','y','treepix','allpix'])

# write python dict to a file
output = open('/home/vagrant/capstone/allhousedat.pkl', 'wb')
pickle.dump(datout, output)
output.close()
