# Detecting houses in aerial images
Aaron Berdanier

## Problem
* Trees grow over buildings and create a risk of property damage - liability for real estate, concern for insurance.
* Homeowners build unpermitted structures on their property - concern for municipalities.

## Solution
Supervised classification can easily distinguish houses from surrounding vegetation (grass, trees).
Detecting buildings automatically in aerial imagery will allow rapid ID of problem structures and properties.

## Data
Using Portland, OR (because they have awesome imagery data)
* Aerial imagery: 

..- 4-channel (R-G-B-Nir) rasters (convert to numpy array)
..- 6-inch resolution over the whole city
..- each approx. 500MB, already downloaded 80 from 2010 (may download repeat scenes from 2012 for change detection)

* House outlines:

..- shapefile, 1.3GB, outlines of each structure
..- >600K buildings, reduced to ~150K 'houses' for analysis

## Workflow
* Bring data into Python
⋅⋅* For each image:
⋅⋅1. Identify structures in scene
⋅⋅⋅⋅* For each structure:
⋅⋅⋅⋅1. Extract raster data for classification
⋅⋅⋅⋅2. Calculate aggregate measures

## Next steps
- [x] Convert example from R to Python
- [x] Download data from USGS and Portland Open Data
- [x] Generate test and training data, demonstrate separation
- [ ] Fit classification model with manual test and training data
- [ ] Classify all houses in an image
- [ ] Figure out how to run images in parallel

## Optional
- [ ] Get tax parcel boundaries (but not open for download right now)
- [ ] Get building permit data for each parcel
- [ ] Get 2012 imagery for change detection
