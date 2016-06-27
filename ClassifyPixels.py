##################################3
from PIL import Image, ImageDraw
from osgeo import gdal, gdalnumeric, ogr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pandas as pa
import re
import seaborn as sea
import shapefile
import simplejson
from sklearn.cross_validation import cross_val_score, KFold
from sklearn import neighbors
from sklearn import metrics
from sklearn.externals import joblib


def array_to_image(a):
    i = Image.fromstring('L',(a.shape[1], a.shape[0]),
        (a.astype('b')).tostring())
    return i

def image_to_array(i):
    a = gdalnumeric.fromstring(i.tostring(), 'b')
    a.shape = i.im.size[1], i.im.size[0]
    return a

def world_to_pixel(geo_matrix, x, y):
    ulX = geo_matrix[0]
    ulY = geo_matrix[3]
    xDist = geo_matrix[1]
    yDist = geo_matrix[5]
    rtnX = geo_matrix[2]
    rtnY = geo_matrix[4]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / xDist)
    return (pixel, line)

def clip_raster(rastarray, geotransform, poly):
    # from http://geospatialpython.com/
    rast = rastarray#rast.ReadAsArray()
    gt = geotransform#rast.GetGeoTransform()
    minX, minY, maxX, maxY = poly.bbox
    # Convert the layer extent to image pixel coordinates
    #minX, maxX, minY, maxY = lyr.GetExtent()
    ulX, ulY = world_to_pixel(gt, minX, maxY)
    lrX, lrY = world_to_pixel(gt, maxX, minY)
    # Calculate the pixel size of the new image
    pxWidth = int(lrX - ulX)
    pxHeight = int(lrY - ulY)
    clip = rast[:, ulY:lrY, ulX:lrX]
    # Create a new geomatrix for the image
    gt2 = list(gt)
    gt2[0] = minX
    gt2[3] = maxY
    # Map points to pixels for drawing the boundary on a blank 8-bit,
    #   black and white, mask image.
    pixels = []
    points = poly.points
    for p in points: pixels.append(world_to_pixel(gt2, p[0], p[1]))
    raster_poly = Image.new('L', (pxWidth, pxHeight), 1)
    rasterize = ImageDraw.Draw(raster_poly)
    rasterize.polygon(pixels, 0) # Fill with zeroes
    mask = image_to_array(raster_poly)
    # Clip the image using the mask
    clip = gdalnumeric.choose(mask, (clip, np.nan)).astype(gdalnumeric.uint8)
    #return (clip, ulX, ulY, gt2)
    return clip

def GetRasExtent(ras): # xmin xmax, ymin, ymax
    tf = ras.GetGeoTransform()
    xL = tf[0]
    yT = tf[3]
    xR = xL + ras.RasterXSize*tf[1] # cols * width
    yB = yT + ras.RasterYSize*tf[5] # rows * height
    return (xL, yB, xR, yT) # xmin, ymin, xmax, ymax

###################
# Bring in raster
fi = '/home/aaron/Desktop/2010448_1N1E08U4BAND/1n1e08u_4band.tif'
ras = gdal.Open(fi)
#ras.GetProjection()
#rasextent = GetRasExtent(ras)
# plot false color image
rasdat = ras.ReadAsArray()
falsecola = np.array([rasdat[3,:,:],rasdat[0,:,:],rasdat[1,:,:]])

#falsecol = np.array([rasdat[3,3000:5000,8000:10000],rasdat[0,3000:5000,8000:10000],rasdat[1,3000:5000,8000:10000]])# subset
falsecola.shape

####### IDENTIFY PIXELS AS HOUSE OR NOT
####### NEED GUI or -X connection
xo = []

def dap(event):
    if event.button == 1:
        if event.xdata is None:
            plt.close(fig)
            print "nonesies"
            xlow = np.random.randint(0,falsecol.T.shape[0]-csz) # random in area
            ylow = np.random.randint(0,falsecol.T.shape[1]-csz) # random in area
            ax.set_xlim([xlow,xlow+csz])
            ax.set_ylim([ylow,ylow+csz])
            fig.show()
        else:
            xo.append(["T",event.xdata,event.ydata])
            print "You added a tree!"
    elif event.button == 3:
        xo.append(["H",event.xdata,event.ydata])
        print "You added a house!"


print "\nSelect trees by left clicking.\nSelect houses by right clicking.\m"
# repeat this process to add new data to set.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(falsecola.T)
cid = fig.canvas.mpl_connect('button_press_event', dap)
#randomly set axis
#xlow = np.random.randint(0,falsecol.T.shape[0]-csz) # random in area
#ylow = np.random.randint(0,falsecol.T.shape[1]-csz) # random in area
#ax.set_xlim([xlow,xlow+csz])
#ax.set_ylim([ylow,ylow+csz])
ax.set_xlim([9100,9500])
ax.set_ylim([4100,4500])
fig.show()

# WHEN DONE
fig.canvas.mpl_disconnect(dap)




dat = pa.DataFrame(xo,columns=['type','x','y'])
dat.x = dat.x.astype(int)
dat.y = dat.y.astype(int)
sr = np.log(rasdat[3]*1./rasdat[0])
trees = sr[dat.x[dat.type=='T'],dat.y[dat.type=='T']]
house = sr[dat.x[dat.type=='H'],dat.y[dat.type=='H']]

bins = np.arange(-2,4,0.1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(trees,bins=bins,alpha=0.6,label="Tree")
ax.hist(house,bins=bins,alpha=0.6,label="House")
ax.set_xlim([-0.5,2.5])
ax.legend(loc='upper right',fontsize=20)
ax.set_xlabel('Spectral index (NIR / Red)',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
fig.show()
