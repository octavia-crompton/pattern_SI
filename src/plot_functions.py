import numpy as np
import pickle
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
from scipy.ndimage import convolve
import pandas as pd
import matplotlib.colors as colors


from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

import cmocean
import matplotlib.pylab as plt
import warnings
warnings.filterwarnings("ignore")

from itertools import cycle
lines = ["-","--","-.",":"]
linecycler = cycle(lines)

plt.style.use('ggplot')
import seaborn as sns
sns.set(font_scale = 1.3)
sns.set_style("whitegrid", {'axes.linewidth': 1.0})
sns.set_context("notebook", font_scale=1.4, rc={"lines.linewidth": 1})
col_list = ["cool blue", "light grey", "viridian", "twilight blue", 
             "dusty purple",  "amber", "greyish", "faded green"]
sns.palplot(sns.xkcd_palette(col_list))
col_list_palette = sns.xkcd_palette(col_list)

plt.rcParams['axes.facecolor']='w'
plt.rcParams['grid.color']= 'grey'
plt.rcParams['grid.alpha']=0.0
plt.rcParams['axes.linewidth']=0.5
plt.rc('axes',edgecolor='grey')

plt.rcParams['axes.spines.top']= 0
plt.rcParams['axes.spines.right']= 0
plt.rcParams['axes.spines.left']= 1
plt.rcParams['axes.spines.bottom']= 1
plt.rc('axes',edgecolor='grey')
plt.rcParams['image.cmap'] = 'viridis'  

import itertools
from itertools import cycle
lines = ["s","o","d","<"]
linecycler = cycle(lines)

def veg_points(isvegc, dx = 1.0, veg_size = 10, ax = '', c = 'g'):
    """
    Creates a scatterplot of a vegetation field
    """
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    xc = np.arange(0, ncol*dx, dx)  + dx/2
    yc = np.arange(0, nrow*dx, dx)  + dx/2
    xc, yc = np.meshgrid(xc, yc)
    
    isvegc = (isvegc*veg_size).astype(float)

    isvegc[isvegc == 0] = np.nan
    if ax == '':      
        fig = plt.figure(figsize = (4,6))
        ax = fig.add_axes()
    else:
        fig = plt.gcf()        
           
    vegplot = ax.scatter(xc+dx/2, yc+dx/2.,
                        s = isvegc.T,
                        c = c,  marker='o', alpha = 0.75)
    
    ax.set_xlim(xc.min(), xc.max())
    ax.set_ylim(yc.min(), yc.max())
    ax.set_xticks([], []);
    ax.set_yticks([], []);

    return vegplot
    

def plot_c(r, dx = 1, ax  = ''):
    """
    """
    import cv2
    thresh = r.astype(np.uint8)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if ax == '':     
        fig, ax = plt.subplots()

    for n, contour in enumerate(contours):
        contour = np.squeeze((contour))

        contours[n] = contour
        if len(contour) >2:
            ax.plot(contour[:, 1]*dx+dx , contour[:, 0]*dx+dx , linewidth=0.7, c= 'k')
            
def plot_c_inv(r, dx = 1, ax  = ''):

    import cv2
    thresh = 1- r.astype(np.uint8)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if ax == '':     
        fig, ax = plt.subplots()

    for n, contour in enumerate(contours):
        contour = np.squeeze((contour))

        contours[n] = contour
        if len(contour) >2:
            ax.plot(contour[:, 1]*dx +dx, contour[:, 0]*dx+dx, linewidth=0.7, c= 'k')


def colormap(df, array,  ax = '',
             colorbar = True, veg_scale = False,
             bounds = '', clabel = '',
             cmin = False, cmax = False,           
             veg_size = 2, plot_veg = False,
             cround = '',cfontsize = 16,
             cmap = cmocean.cm.deep):

    
    isvegc = df['isvegc'].astype(float)
    dx = df['dx']
    
    ncol = array.shape[0]
    nrow = array.shape[1]

    xc = np.arange(0, ncol*dx, dx)  + dx/2
    yc = np.arange(0, nrow*dx, dx)  + dx/2
    xc, yc = np.meshgrid(xc, yc)
    

    xc = xc.T
    yc = yc.T
    
    if isvegc.sum() == 0:
      veg_scale = False  
      
    if ax == '':      
        fig = plt.figure(figsize = (5,5))
        ax = fig.add_subplot(111)
    else:
        fig = plt.gcf()
  
    if bounds == '':
    
        if veg_scale == True:
            scale_vals = array[isvegc == True].ravel()
        else: 
            scale_vals =  array.ravel()
        
        if type(cmin) == bool:            
            cmin = np.nanmin(scale_vals)

        if type(cmax) == bool:            
            cmax = np.nanmax(scale_vals)

        bounds = np.linspace(cmin, cmax, 100)

        if cround != '':
            cmin = np.round(cmin, cround)
            bounds = np.arange(cmin, cmax, 1/10.**cround)
            
        if np.sum(array.astype(int) - array) == 0:
             bounds = np.arange(cmin, cmax+1.1,1)
            
        if np.nanstd(array) < 1e-5:
            bounds = np.linspace(cmin-1, cmax+1, 10)


    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)    

    zinflplot = ax.pcolormesh( xc.T, yc.T, array.T,
                           norm = norm,
                           cmap=cmap, alpha= 1);
                                               
    if colorbar == True:
      from mpl_toolkits.axes_grid1 import make_axes_locatable
    
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)

      cbh = fig.colorbar(zinflplot,cax = cax,shrink=1)
      cbh.set_label(clabel, fontsize = cfontsize)
      cbh.ax.tick_params(labelsize=cfontsize) 
      cbh.ax.locator_params(nbins=5)

    if plot_veg == True:
      vegplot = ax.scatter(xc+df.dx/2., yc+df.dx/2.,
                          s = isvegc*veg_size,
                          c = 'g',                        
                          marker='o', alpha = 0.75)

    ax.set_ylim(yc.min(), yc.max())
    ax.set_xlim(xc.min(), xc.max())
    ax.set_xticks([], []);
    ax.set_yticks([], []);
    
    return zinflplot

def triptych(sim):
    """
    Creates a side-by-side plot of vegetation distribution, infiltration depth, 
    and maximum flow velocity Umax
    """
    fig = plt.figure(figsize= (14,6))
    plt.subplots_adjust(wspace = 0.3)

    for i, label in enumerate(('A', 'B', 'C')):
        ax = plt.subplot(1,3,i+1)
        ax.text(-0.05, 1.08, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')

    ax1 = plt.subplot(131)
    veg_points(sim.isvegc, dx  = sim.dx, ax = ax1)

    ax1 = plt.subplot(132)
    zinflplot = colormap(sim,sim['zinflc'], ax = ax1,
                         clabel= '$I$ (cm)', colorbar = True , cround = 1)

    ax1 = plt.subplot(133)
    zinflplot = colormap(sim,sim['vmax'], ax = ax1, clabel= 'velocity (cm/s)',
                         colorbar = True, cmap = "Blues",
                        cround = 1,     veg_scale=False)



def plot_3D_veg(sim): 
    """
    3D plot of the vegetation field
    """
    fig = plt.figure( figsize = (15, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Get rid of colored axes planes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticks([], []);
    ax.set_zticks([], []);
    ax.set_yticks([], []);
    # plt.axis('off')
    ax.grid(False)

    isvegc = sim.isvegc
    dx = sim.dx
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    xc = np.arange(0, ncol*dx, dx)  + dx/2
    yc = np.arange(0, nrow*dx, dx)  + dx/2
    xc, yc = np.meshgrid(xc, yc)

    xc = xc.T
    yc = yc.T

    #Plot the surface with face colors taken from the array we made.
    norm = plt.Normalize()
    colors = cm.Greens(norm(sim.isvegc ))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    im = ax.scatter(xc[ sim.isvegc == 1], yc[ sim.isvegc == 1],
            yc[ sim.isvegc == 1], c = 'g',  marker='o',  s = 20, alpha =1)

    ax.view_init(20, 195)


def plot_3D_zinflc(sim):
    """
    3D plot of the infiltration field map

    """
    isvegc = sim.isvegc
    dx = sim.dx
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    xc = np.arange(0, ncol*dx, dx)  + dx/2
    yc = np.arange(0, nrow*dx, dx)  + dx/2
    xc, yc = np.meshgrid(xc, yc)


    xc = xc.T
    yc = yc.T

    fig = plt.figure( figsize = (10, 5))
    ax = fig.add_subplot(111, projection='3d')
    # Get rid of colored axes planes`
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticks([], []);
    ax.set_zticks([], []);
    ax.set_yticks([], []);

    ax.grid(False)


    # # Plot the surface with face colors taken from the array we made.
    norm = plt.Normalize()
    colors = cmocean.cm.deep(norm(sim.zinflc ))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    im = ax.plot_surface(xc, yc+1 ,yc, facecolors = colors , rstride = 1, cstride = 1,
                           linewidth=0,antialiased=True, shade=False)

    ax.view_init(25, 195)
    return im

def quick_hydro(sims):
    """
    """
    plt.figure()
    try:
        for key in sims.keys():
            sim = sims[key]
            t_h = np.arange(len(sim['hydro']))
            plt.plot(t_h/60., sim['hydro']*3.6e3)
    except:
        for key in sims.index:
            sim = sims.loc[key]
            t_h = np.arange(len(sim['hydro']))
            plt.plot(t_h/60., sim['hydro']*3.6e3)           
    plt.xlabel('time (minutes)')
    plt.ylabel('cm/hr')

