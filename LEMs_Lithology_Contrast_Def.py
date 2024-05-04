# -*- coding: utf-8 -*-
"""
CSDMS 2024

@author: Renato da Silva
"""

# Code block 1 - importing libraries

import numpy as np
from matplotlib import pyplot as plt

from landlab import RasterModelGrid, imshow_grid
from landlab.components import (
    ChannelProfiler,
    ChiFinder,
    FlowAccumulator,
    SteepnessFinder,
    StreamPowerEroder,
    LinearDiffuser,
)

from landlab.io import write_esri_ascii

# Code Block 2 - Make a grid and set boundary conditions

number_of_rows = 50  # number of raster cells in vertical direction (y)
number_of_columns = 100  # number of raster cells in horizontal direction (x)
dxy = 200  # side length of a raster model cell, or resolution [m]

# Below is a raster (square cells) grid, with equal width and height
mg1 = RasterModelGrid((number_of_rows, number_of_columns), dxy)

# Set boundary conditions - only the south side of the grid is open.
# Boolean parameters are sent to function in order of
# east, north, west, south.
mg1.set_closed_boundaries_at_grid_edges(True, True, True, False)

# Code Block 3 - initial grid of elevation of zeros with a very small amount of noise

np.random.seed(35)  # seed set so our figures are reproducible
mg1_noise = (np.random.rand(mg1.number_of_nodes) / 1000.0
             )  # initial noise on elevation grid

# set up the elevation on the grid
z1 = mg1.add_zeros("topographic__elevation", at="node")
z1 += mg1_noise

# Code Block 4 - Set parameters related to time

tmax = 1e5  # time for the model to run [yr] (Original value was 5E5 yr)
dt = 1000  # time step [yr] (Original value was 100 yr)
total_time = 0  # amount of time the landscape has evolved [yr]
# total_time will increase as you keep running the code.

t = np.arange(0, tmax, dt)  # each of the time steps that the code will run

# Code Block 5 - Set parameters for incision and intializing all of the process components

# Creating function create_lithology  
def create_lithology(number_of_rows,number_of_columns, K_sp1, K_sp2):
    """
    Create a lithology array to be used on the StreamPowerEroder function

    Return lithology
    """
    lithology = np.zeros((number_of_rows, number_of_columns))
    
    for row,column in enumerate(lithology):
        if row<=9:
            for i in range(number_of_columns):
                lithology[row,i]=K_sp1
        else:
            for i in range(number_of_columns):
                lithology[row,i]=K_sp2
    return lithology

# Creating lithology variable using create_lithology function
lithology = create_lithology(number_of_rows,number_of_columns,1.0e-5,5.0e-5)

# Original K_sp value is 1e-5
K_sp = 1.0e-5  # units vary depending on m_sp and n_sp
m_sp = 0.5  # exponent on drainage area in stream power equation
n_sp = 1.0  # exponent on slope in stream power equation

frr = FlowAccumulator(mg1)  # intializing flow routing
spr = StreamPowerEroder(mg1, K_sp=lithology, m_sp=m_sp, n_sp=n_sp,
                        threshold_sp=0.0)  # initializing stream power incision
ld = LinearDiffuser(mg1, linear_diffusivity=0.0025)

theta = m_sp / n_sp
# initialize the component that will calculate channel steepness
sf = SteepnessFinder(mg1, reference_concavity=theta, min_drainage_area=1000.0)
# initialize the component that will calculate the chi index
cf = ChiFinder(mg1,
               min_drainage_area=1000.0,
               reference_concavity=theta,
               use_true_dx=True)

# Code Block 6 - Initialize rock uplift rate and running until topographic equilibrium

#  uplift_rate [m/yr] (Original value is 0.0001 m/yr)
uplift_rate = np.ones(mg1.number_of_nodes) * 0.0001

# Running simulation until topographic equilibrium (for 300 Ma)

def running_equilibrium(mg, z, fa, sp, ld, uplift_rate, dt, mg1_noise):
    """
    Run the model until topography reaches the topographic equilibrium.

    Parameters:
    - mg: the grid representing the landscape
    - z: array of elevation values
    - fa: flow accumulator component
    - sp: stream power erosion component
    - ld: linear diffusion component
    - uplift_rate: array of uplift rates
    - dt: time step
    - mg1_noise: noise level in the grid used as a threshold for equilibrium

    Returns:
    - mg: the grid in equilibrium
    """
    zdiffm = 1  # Initial difference must be set to a value greater than any noise threshold

    while zdiffm > np.max(mg1_noise):
        zinit = z.copy()
    
        # Apply uplift
        mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_rate[mg.core_nodes] * dt
    
        # Run the landscape evolution processes
        fa.run_one_step()  # flow accumulator
        sp.run_one_step(dt) # stream power
        ld.run_one_step(dt) # linear diffuser
        
        # Calculate the maximum difference in elevation from the previous step
        zdiffm = np.max(np.abs(zinit - z))

    return mg

mg1 = running_equilibrium(mg1,z1,frr,spr,ld,uplift_rate,dt,mg1_noise)

# Code Block 7 - for loop landscape evolution

for ti in t:
    z1[mg1.
       core_nodes] += uplift_rate[mg1.core_nodes] * dt  # uplift the landscape
    frr.run_one_step()  # route flow
    spr.run_one_step(dt)  # fluvial incision
    ld.run_one_step(dt) # linear diffusion
    total_time += dt  # update time keeper
    print(total_time)
    
# Code Block 8 - plot figure

imshow_grid(mg1,
            "topographic__elevation",
            grid_units=("m", "m"),
            var_name="Elevation (m)")
title_text = f"$K_{{sp}}$={K_sp}; $time$={total_time} yr; $dx$={dxy} m"
plt.title(title_text)

max_elev = np.max(z1)
print("Maximum elevation is ", np.max(z1))   

# Code Block 9 - Plot the slope and area data at each point on the landscape (in log-log space)

# Extracting drainage area and slope
drainage_area = mg1.at_node["drainage_area"][mg1.core_nodes]
steepest_slope = mg1.at_node["topographic__steepest_slope"][mg1.core_nodes]

# Log-transform the data
log_drainage_area = np.log(drainage_area)
log_slope = np.log(steepest_slope)

# Perform linear regression on the log-transformed data
coefficients = np.polyfit(log_drainage_area, log_slope, 1)
p = np.poly1d(coefficients)  # Create a polynomial function with the fit coefficients

# Calculate the R^2 value
slope_predicted = p(log_drainage_area)
r_squared = np.corrcoef(log_slope, slope_predicted)[0, 1] ** 2

# Plot the log-log graph
plt.loglog(drainage_area, steepest_slope, "b.", label='Data')

# Plot the regression line
regression_line = np.exp(slope_predicted)
plt.plot(drainage_area, regression_line, 'r-', label=f'Best Fit: y={coefficients[0]:.2f}x + {coefficients[1]:.2f}, $R^2$={r_squared:.2f}')

# Add labels and title
plt.ylabel("Topographic slope")
plt.xlabel("Drainage area (m^2)")
title_text = f"$K_{{sp}}$={K_sp}; $time$={total_time} yr; $dx$={dxy} m"
plt.title(title_text)

# Add a legend
plt.legend()

# Display the plot
plt.show()

# Code Block 10

# profile the largest channels, set initially to find the mainstem channel in the three biggest watersheds
# you can change the number of watersheds, or choose to plot all the channel segments in the watershed that
# have drainage area below the threshold (here we have set the threshold to the area of a grid cell).
prf = ChannelProfiler(mg1,
                      number_of_watersheds=20,
                      main_channel_only=True,
                      minimum_channel_threshold=dxy**2)
prf.run_one_step()

# plot the elevation as a function of distance upstream
plt.figure(1)
title_text = f"$K_{{sp}}$={K_sp}; $time$={total_time} yr; $dx$={dxy} m"
prf.plot_profiles(xlabel='distance upstream (m)',
                  ylabel='elevation (m)',
                  title=title_text)

# plot the location of the channels in map view
plt.figure(2)
prf.plot_profiles_in_map_view()

# slope-area data in just the profiled channels
plt.figure(3)
for i, outlet_id in enumerate(prf.data_structure):
    for j, segment_id in enumerate(prf.data_structure[outlet_id]):
        if j == 0:
            label = "channel {i}".format(i=i + 1)
        else:
            label = '_nolegend_'
        segment = prf.data_structure[outlet_id][segment_id]
        profile_ids = segment["ids"]
        color = segment["color"]
        plt.loglog(
            mg1.at_node["drainage_area"][profile_ids],
            mg1.at_node["topographic__steepest_slope"][profile_ids],
            '.',
            color=color,
            label=label,
        )

plt.legend(loc="lower left")
plt.xlabel("drainage area (m^2)")
plt.ylabel("channel slope [m/m]")
title_text = f"$K_{{sp}}$={K_sp}; $time$={total_time} yr; $dx$={dxy} m"
plt.title(title_text)

# Code Block 11

# calculate the chi index
cf.calculate_chi()

# chi-elevation plots in the profiled channels
plt.figure(4)

for i, outlet_id in enumerate(prf.data_structure):
    for j, segment_id in enumerate(prf.data_structure[outlet_id]):
        if j == 0:
            label = "channel {i}".format(i=i + 1)
        else:
            label = '_nolegend_'
        segment = prf.data_structure[outlet_id][segment_id]
        profile_ids = segment["ids"]
        color = segment["color"]
        plt.plot(
            mg1.at_node["channel__chi_index"][profile_ids],
            mg1.at_node["topographic__elevation"][profile_ids],
            color=color,
            label=label,
        )

plt.xlabel("chi index (m)")
plt.ylabel("elevation (m)")
plt.legend(loc="lower right")
title_text = f"$K_{{sp}}$={K_sp}; $time$={total_time} yr; $dx$={dxy} m; concavity={theta}"
plt.title(title_text)

# chi map
plt.figure(5)
imshow_grid(
    mg1,
    "channel__chi_index",
    grid_units=("m", "m"),
    var_name="Chi index (m)",
    cmap="jet",
)
title_text = f"$K_{{sp}}$={K_sp}; $time$={total_time} yr; $dx$={dxy} m; concavity={theta}"
plt.title(title_text)

# Code Block 12

# calculate channel steepness
sf.calculate_steepnesses()

# plots of steepnes vs. distance upstream in the profiled channels
plt.figure(6)

for i, outlet_id in enumerate(prf.data_structure):
    for j, segment_id in enumerate(prf.data_structure[outlet_id]):
        if j == 0:
            label = "channel {i}".format(i=i + 1)
        else:
            label = '_nolegend_'
        segment = prf.data_structure[outlet_id][segment_id]
        profile_ids = segment["ids"]
        distance_upstream = segment["distances"]
        color = segment["color"]
        plt.plot(
            distance_upstream,
            mg1.at_node["channel__steepness_index"][profile_ids],
            'x',
            color=color,
            label=label,
        )

plt.xlabel("distance upstream (m)")
plt.ylabel("steepness index")
plt.legend(loc="upper left")
plt.title(
    f"$K_{{sp}}$={K_sp}; $time$={total_time} yr; $dx$={dxy} m; concavity={theta}"
)

# channel steepness map
plt.figure(7)
imshow_grid(
    mg1,
    "channel__steepness_index",
    grid_units=("m", "m"),
    var_name="Steepness index ",
    cmap="jet",
)
title_text = ("$K_{sp}$=" + str(K_sp) + "; $time$=" + str(total_time) +
              "yr; $dx$=" + str(dxy) + "m" + "; concavity=" + str(theta))
plt.title(
    f"$K_{{sp}}$={K_sp}; $time$={total_time} yr; $dx$={dxy} m; concavity={theta}"
)

# Code Block 13 - Export data

write_file_name = 'C:/Users/renat/Documents/GitHub/LandLab_LEMs_Lithology/outputs_txt/Data_file_300k_ky_'

write_esri_ascii(write_file_name + 'topographic__elevation.txt', mg1, 'topographic__elevation')
write_esri_ascii(write_file_name + 'drainage_area.txt', mg1, 'drainage_area')
write_esri_ascii(write_file_name + 'topographic__steepest_slope.txt', mg1, 'topographic__steepest_slope')
write_esri_ascii(write_file_name + 'channel__steepness_index.txt', mg1, 'channel__steepness_index')
write_esri_ascii(write_file_name + 'channel__chi_index.txt', mg1, 'channel__chi_index')

# Code Block 14 - Assessing evolution of parameters through time

# provavelmente consigo fazer via matlab... eu importo esses arquivos como gridobjs e calculo
# a media de cada um deles para a bacia (usando o outlet, por ex). Provavelmente consigo 
# fazer isso importando a topografia e rodando makestreams, basinpicker, processriverbasins, 
# e compileriverbasins