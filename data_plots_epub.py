# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:28:43 2024

@author: renat
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('C:\\Users\\renat\\Documents\\GitHub\\CSDMS_RDasilva_EPub_Lithology\\CSDMS_sheet.csv')

# Group data by 'ID' and 'Time [ky]', then unstack 'Time [ky]' to make it columns
df_grouped = df.set_index(['ID', 'Time [ky]']).unstack(level='Time [ky]')

# Basins to select
basin_indices = [5, 7, 8]

# Variables of interest
vars_interest = ['drainage_area', 'mean_el', 'mean_ksn', 'mean_gradient']

# Initialize the plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Titles and y-labels for each subplot
titles = ['Drainage Area vs. Time', 'Mean Elevation vs. Time', 'Mean ksn vs. Time', 'Mean Gradient vs. Time']
ylabels = ['Drainage Area (km^2)', 'Mean Elevation (m)', 'Mean ksn (dimensionless)', 'Mean Gradient (dimensionless)']

# Times for plotting and annotation
time_points = [0, 100, 200, 300]

# Function to dynamically adjust text offsets based on value changes
def adjust_offsets(values):
    offsets = np.zeros(len(values))
    for i in range(1, len(values)):
        if abs(values[i] - values[i-1]) < 0.1:  # threshold for determining if values are too close
            offsets[i] = offsets[i-1] + 5  # adjust offset upward if too close
        else:
            offsets[i] = 0  # reset offset if values are far apart
    return offsets

# Plot each variable
for i, var in enumerate(vars_interest):
    for basin in basin_indices:
        data_to_plot = df_grouped[var].loc[basin].values.flatten()
        axes[i].plot(time_points, data_to_plot, label=f'Basin {basin}')
        # Adjust annotations
        offsets = adjust_offsets(data_to_plot)
        for j, (x, y) in enumerate(zip(time_points, data_to_plot)):
            axes[i].text(x, y + offsets[j], f'{y:.2f}', ha='center', va='bottom')
    axes[i].set_title(titles[i])
    axes[i].set_xlabel('Time (ky)')
    axes[i].set_ylabel(ylabels[i])
    axes[i].legend(loc='best')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()