# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:32:44 2024

@author: renat
"""
# Part 1 - Introduction of Landlab grids

from landlab import RasterModelGrid

# Creating a new grid
mg = RasterModelGrid((10,40),5) # ((rows, columns), spacing*) *dx

mg.number_of_node_columns # 40 columns
mg.number_of_nodes # 10x40 = 400 node cells

# Adding a new attribute/data field to this grid
z = mg.add_zeros('elevation', at='node') # here mg is an array of zeros
z.size # or len(z), which is 400 

# Accessing a data value
mg.at_node['elevation'][5] = 1000 # accessing the sixth element value

mg['node']['elevation'][5]
mg.at_node['elevation'][5]

# Accessing mg's keys
mg.at_node.keys() # ['elevation]

# So each attribute works as a dictionary, with pairs of data and keys

# add_field function
import numpy as np
elevs_in = np.random.rand(mg.number_of_nodes)

mg.add_field(
    'elevation', elevs_in, at='node', units='m', copy=True, clobber=True
    )

core_node_elevs = mg.at_node['elevation'][mg.core_nodes] # accessing values

# add_ones function
veg = mg.add_ones('percent_vegetation', at='cell')
mg.at_cell.keys() # ['percent_vegetation]

mg.at_cell['percent_vegetation'].size # 304

# Field Initialization
mg.add_empty('name', at='group', units='-') # group can be any of 'node','link','cell','face','corner','junction','patch'
mg.add_ones('name', at='group', units='-')
mg.add_zeros('name', at='group', units='-')

# Field Creation from Existing Data
value_array = [] # supposed that this has data
mg.add_field('name', value_array, at='group', units='-', copy=False, clobber=True)

# Field Access
mg.at_node # or mg['node']
mg.at_corner # or mg['corner']
mg.at_cell # or mg['cell']
mg.at_face # or mg['face']
mg.at_link # or mg['link']
mg.at_patch # or mg['patch']

# Each of this is followed by the field name. Take as an example the field 'elevation':
mg.at_node['elevation'] # or
mg['node']['elevation']    

# Representing Gradients in a Landlab Grid
gradient = mg.calculate_gradients_at_active_links(z)
gradient = mg.calculate_slope_aspect_at_nodes_burrough(vals='elevation')

# Managing Grid Boundaries
mg.status_at_node # return type of boundary condition of each node in the grid

# Types
mg.BC_NODE_IS_CORE # Type 0
mg.BC_NODE_IS_FIXED_VALUE # Type 1
mg.BC_NODE_IS_FIXED_GRADIENT # Type 2
mg.BC_NODE_IS_LOOPED # Type 3, used for looped boundaries
mg.BC_NODE_IS_CLOSED # Type 4

# Method to interact with (i.e., set and update BC)
mg.set_closed_boundaries_at_grid_edges(right, top, left, bottom)
mg.set_fixed_value_boundaries_at_grid_edges(right, top, left, bottom)
mg.set_looped_boundaries(top_bottom_are_looped, sides_are_looped)

# Example
grid = RasterModelGrid((5,5))

grid.set_closed_boundaries_at_grid_edges(False, True, False, True)
grid.number_of_active_links # 18
grid.status_at_node.reshape((5, 5)) 

grid.status_at_node[[6, 8]] = grid.BC_NODE_IS_CLOSED
grid.status_at_node.reshape((5, 5)) # note the 6 and 8 position

grid.number_of_active_links # 12

# Part 1.1 - Importing DEMs

# read_esri_ascii
from landlab.io import read_esri_ascii
(mg,z) = read_esri_ascii('MyArcOutput.txt', name='elevation')
mg.at_node.keys() # ['elevation']

# read_netcdf
from landlab.io.netcdf import read_netcdf
mg = read_netcdf('myNetCdf.nc')

# Part 1.2 - Plotting and Visualizing

# visualizing a grid
from landlab import imshow_grid
from pylab import show, figure

mg = RasterModelGrid((50, 50), 1.0) # make a grid to plot
z = mg.node_x*0.1 # make a sloping surface

mg.add_field(
    'elevation', z, at='node', units='meters', copy=True
    ) 

figure('Elevations from the field') # new fig, with a name
imshow_grid(mg, 'elevation')

figure(
       'You can also use values directly, instead of fields'
       )
imshow_grid(mg, z)
show()

# Visualizing transects
from pylab import show, plot
mg = RasterModelGrid((10, 10), 1.0)
z = mg.node_x*0.1

my_section = mg.node_vector_to_raster(z, flip_vertically=True)[:, 5]
my_ycoords = mg.node_vector_to_raster(mg.node_y, flip_vertically=True)[:, 5]

plot(my_ycoords, my_section)
show()

# Part 2 - The Component Library

# Part 3 - What goes into a LandLab Model?


















