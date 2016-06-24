##################################3
####################################
########################
from PIL import Image, ImageDraw
from osgeo import gdal, gdalnumeric, ogr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pandas as pa
import pickle
import re
import seaborn as sea
import shapefile

def array_to_image(a):
    i = Image.fromstring('L',(a.shape[1], a.shape[0]),(a.astype('b')).tostring())
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
    return clip

def GetRasExtent(ras): # xmin xmax, ymin, ymax
    tf = ras.GetGeoTransform()
    xL = tf[0]
    yT = tf[3]
    xR = xL + ras.RasterXSize*tf[1] # cols * width
    yB = yT + ras.RasterYSize*tf[5] # rows * height
    return (xL, yB, xR, yT) # xmin, ymin, xmax, ymax

c = shapefile.Reader('/home/vagrant/PortlandBuildings/Building_Footprints_pdx')
#c.bbox # bounding box of the shapefile
records = c.records() # get the 'data' from the shapefile
#len(records) # number of buildings in Portland

recpa = pa.DataFrame(records, columns=pa.DataFrame(c.fields).ix[1:,0])
#recpa.head()
recpa.BLDG_SQFT = pa.to_numeric(recpa.BLDG_SQFT,'coerce') # force to numeric - necessary?
selection = recpa.index[(recpa.BLDG_USE.isin(("Single Family Residential","Multi Family Residential")))
                        & (recpa.BLDG_SQFT>700)& (recpa.BLDG_SQFT<7000)] # exclude sheds and mansions
#len(selection) # number of houses
#plt.hist(recpa.BLDG_SQFT[selection],bins=range(0,int(max(recpa.BLDG_SQFT[selection])),1000))

###################
# Bring in raster
d = "/home/vagrant/rast/"
dirs = os.listdir(d)

dir2 = os.listdir('/home/vagrant/capstone/outdat/')

dirs = [di for di in dirs if di+'_housedat.pkl' not in dir2]

# for each raster directory....
for filedir in dirs:
    ff = os.listdir(d+filedir)
    ra = [x for x in ff if re.match(".+tif",x)]
    fi = d+filedir+"/"+ra[0]
    ras = gdal.Open(fi)
    rx = GetRasExtent(ras)
    rasdat = ras.ReadAsArray()
    print filedir

    # make new shape with only houses in raster
    w = shapefile.Writer(c.shapeType)
    for s in selection:
        css = c.shape(s)
        bbox = css.bbox
        if bbox[0] < rx[0]: continue # minx
        elif bbox[1] < rx[1]: continue # miny
        elif bbox[2] > rx[2]: continue # maxx
        elif bbox[3] > rx[3]: continue #maxy
        w._shapes.append(css)

    sh = w.shapes()

    ra = ras.ReadAsArray()
    gt = ras.GetGeoTransform()

    outdat = []
    cnt = 1
    for house in sh:
        try:
            ras2 = clip_raster(ra,gt,house)
        except:
            continue
        sr = ras2[3]*1./ras2[0]
        sr = sr[np.isfinite(sr)]
        sr = sr.tolist()
        outdat.append({'corner':house.points[0], 'sr':sr}) # corner coordinate
        cnt += 1
        if cnt%100 == 0: print cnt

    print 'done'
    # write python dict to a file
    output = open('/home/vagrant/capstone/outdat/'+filedir+'_housedat.pkl', 'wb')
    pickle.dump(outdat, output)
    output.close()
