#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue October 11 09:54:00 2022
    Copyright (C) <2022>  <Håvard Boutera Toft>
    htla@nve.no

    This python script reimplements the Potential Release Area proposed 
    by Veitinger et al. (2016) and modified by Sharp et al., (2018) 
    using python libraries.

    References:
    https://github.com/jocha81/Avalanche-release
        Veitinger, J., Purves, R. S., & Sovilla, B. (2016). Potential 
    slab avalanche release area identification from estimated winter 
    terrain: a multi-scale, fuzzy logic approach. Natural Hazards and 
    Earth System Sciences, 16(10), 2211-2225.
        Sharp, A. E. A. (2018). Evaluating the Exposure of Heliskiing 
    Ski Guides to Avalanche Terrain Using a Fuzzy Logic Avalanche 
    Susceptibility Model. University of Leeds: Leeds, UK.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# import standard libraries
import numpy as np
import rasterio, rasterio.mask
from osgeo import gdal
import os
from numpy.lib.stride_tricks import as_strided
from collections import deque
import sys

def PRA(DEM, FOREST, radius, prob, winddir, windtol, forest_type):
    # Check if path exits
    if os.path.exists(DEM) is False:
        print("The path {} does not exist".format(DEM))

    if forest_type in ['pcc', 'stems']:
        # Check if path exits
        if os.path.exists(FOREST) is False:
            print("The path {} does not exist".format(FOREST))
        
        print(DEM, FOREST, radius, prob, winddir, windtol, forest_type)

    if forest_type in ['no_forest']:
        print(DEM, radius, prob, winddir, windtol, forest_type)

    #########################
    # --- Define functions
    #########################

    def sliding_window_view(arr, window_shape, steps):
        """ 
        Produce a view from a sliding, striding window over `arr`.
        The window is only placed in 'valid' positions - no overlapping
        over the boundary
        """
        
        in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]
        window_shape = np.array(window_shape)  # [Wx, (...), Wz]
        steps = np.array(steps)  # [Sx, (...), Sz]
        nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

        # number of per-byte steps to take to fill window
        window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
        # number of per-byte steps to take to place window
        step_strides = tuple(window_strides[-len(steps):] * steps)
        # number of bytes to step to populate sliding window view
        strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

        outshape = tuple((in_shape - window_shape) // steps + 1)
        # outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
        outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
        return as_strided(arr, shape=outshape, strides=strides, writeable=False)

    def sector_mask(shape,centre,radius,angle_range): # used in windshelter_prep
        """
        Return a boolean mask for a circular sector. The start/stop angles in  
        `angle_range` should be given in clockwise order.
        """

        x,y = np.ogrid[:shape[0],:shape[1]]
        cx,cy = centre
        tmin,tmax = np.deg2rad(angle_range)

        # ensure stop angle > start angle
        if tmax < tmin:
                tmax += 2*np.pi

        # convert cartesian --> polar coordinates
        r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
        theta = np.arctan2(x-cx,y-cy) - tmin

        # wrap angles between 0 and 2*pi
        theta %= (2*np.pi)

        # circular mask
        circmask = r2 <= radius*radius

        # angular mask
        anglemask = theta <= (tmax-tmin)
        
        a = circmask*anglemask

        return a

    def windshelter_prep(radius, direction, tolerance, cellsize):
        x_size = y_size = 2*radius+1
        x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
        cell_center = (radius, radius)
        dist = (np.sqrt((x_arr - cell_center[0])**2 + (y_arr - cell_center[1])**2))*cellsize
        # dist = np.round(dist, 5)
        
        mask = sector_mask(dist.shape, (radius, radius), radius, (direction, tolerance))
        mask[radius, radius] = True # bug fix
        
        return dist, mask

    def windshelter(x, prob, dist, mask, radius): # applying the windshelter function
        data = x*mask
        data[data==profile['nodata']]=np.nan
        data[data==0]=np.nan
        center = data[radius, radius]
        data[radius, radius]=np.nan
        data = np.arctan((data-center)/dist)
        data = np.nanquantile(data, prob)
        return data

    def windshelter_window(radius, prob):

        dist, mask = windshelter_prep(radius, winddir - windtol + 270, winddir + windtol + 270, cell_size)
        window = sliding_window_view(array[-1], ((radius*2)+1,(radius*2)+1), (1, 1))

        nc = window.shape[0]
        nr = window.shape[1]
        ws = deque()

        for i in range(nc):
            for j in range(nr):
                data = window[i, j]
                data = windshelter(data, prob, dist, mask, radius).tolist()
                ws.append(data)

        data = np.array(ws)
        data = data.reshape(nc, nr)
        data = np.pad(data, pad_width=radius, mode='constant', constant_values=profile['nodata'])
        data = data.reshape(1, data.shape[0], data.shape[1])
        data = data.astype('float32')
        
        profile.update({"dtype": "float32"})
        
        # Save raster to path using meta data from dem.tif (i.e. projection)
        with rasterio.open('windshelter.tif', "w", **profile) as dest:
            dest.write(data)

    ########################################
    # --- Calculate slope and windshelter
    ########################################

    # Calculate slope
    gdal.DEMProcessing('slope.tif', DEM, "slope", computeEdges=True)

    # Calculate windshelter
    with rasterio.open(DEM) as src:
        array = src.read()
        array = array.astype('float')
        profile = src.profile
        cell_size = profile['transform'][0]
    data = windshelter_window(radius, prob)

    #######################
    # --- Fuzzy operator
    #######################

    np.seterr(all='ignore')

    # --- Define bell curve parameters for slope
    a = 11
    b = 4
    c = 43

    with rasterio.open("slope.tif") as src:
        slope = src.read()
        profile = src.profile

    slopeC = 1/(1+((slope-c)/a)**(2*b))

    # --- Define bell curve parameters for windshelter
    a = 2
    b = 5
    c = 2

    with rasterio.open("windshelter.tif") as src:
        windshelter = src.read()
        
    windshelterC = 1/(1+((windshelter-c)/a)**(2*b))

    # --- Define bell curve parameters for forest stem density
    if forest_type in ['stems']:
        a = 350
        b = 2.5
        c = -120

    # --- Define bell curve parameters for percent canopy cover
    if forest_type in ['pcc', 'no_forest']:
        a = 350
        b = 2.5
        c = -120

    if forest_type in ['pcc', 'stems']:
        with rasterio.open(FOREST) as src:
            forest = src.read()
            
    if forest_type in ['no_forest']:
        with rasterio.open(DEM) as src:
            forest = src.read()
            forest = np.where(forest > -100, 0, forest)

    forestC = 1/(1+((forest-c)/a)**(2*b))
    forestC[np.where(forestC <= 0)] = 1

    # --- Fuzzy logic operator
    minvar = np.minimum(slopeC, windshelterC)
    minvar = np.minimum(minvar, forestC)

    PRA = (1-minvar)*minvar+minvar*(slopeC+windshelterC+forestC)/3

    PRA = PRA.astype('float32')
    PRA[np.where(PRA <= 0)] = 0

    # --- Update metadata
    profile.update({'dtype': 'float32', 'nodata': 0})

    # --- Save raster to path using meta data from dem.tif (i.e. projection)
    with rasterio.open('PRA.tif', "w", **profile) as dest:
        dest.write(PRA)

    print('PRA complete')

if __name__ == "__main__":
    DEM = sys.argv[1]
    FOREST = sys.argv[2]
    radius = int(sys.argv[3])
    prob = float(sys.argv[4])
    winddir = int(sys.argv[5])
    windtol = int(sys.argv[6])
    forest_type = str(sys.argv[7])
    PRA(DEM, FOREST, radius, prob, winddir, windtol, forest_type)
