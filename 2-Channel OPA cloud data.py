# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:01:11 2023

@author: limyu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:08:57 2023

@author: limyu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy import interpolate
from scipy.signal import chirp, find_peaks, peak_widths

max_column_index = []
photon_dataframe_left = pd.DataFrame([])
photon_dataframe_right = pd.DataFrame([])
max_values_left = []
max_values_right = []
verticle_peaks_left = []
verticle_peaks_right = []
voltage = ["0V","2V","4V", "6V", "8V","10V","12V","14V","16V","17V"]
for v in voltage:
    df = pd.read_csv("https://raw.githubusercontent.com/yd145763/OPA_data_analysis/main/"+v+"%20optimized%201092%20nm%202%20arms%20OPA.csv")
    df=df.dropna(axis=1)
    x = np.linspace(0, 9570, num=320)
    x = x/22.5
    y = np.linspace(0, 7650, num=256)
    y = y/22.5
    
    df1 = df.iloc[100:136, 107:167]
    df1 = df1.reset_index(drop=True)
    df1.columns = range(df1.shape[1])
    x1 = np.linspace(143, 222, num=60)
    y1 = np.linspace(133, 181, num=36)    

    
    df1 = pd.DataFrame(df1)
    df2 = df1.iloc[:,48:]
    # Find the maximum value in the DataFrame
    max_value = df2.values.max()
    max_values_right.append(max_value)
    # Find the column name where the maximum value belongs to
    mask = df1.isin([max_value])
    cols = mask.any()[mask.any()].index.tolist()
    col = cols[-1]
    max_column_index.append(cols)
    
    count = df1.iloc[:, col]
    tck = interpolate.splrep(y1, count, s=2, k=4) 
    x_new = np.linspace(min(y1), max(y1), 1000)
    y_fit = interpolate.BSpline(*tck)(x_new)

    photon_dataframe_right[v] = y_fit
    peak_index = np.argmax(y_fit)
    peak_position = x_new[peak_index]
    verticle_peaks_right.append(peak_position)
    
for v in voltage:
    df = pd.read_csv("https://raw.githubusercontent.com/yd145763/OPA_data_analysis/main/"+v+"%20optimized%201092%20nm%202%20arms%20OPA.csv")
    df=df.dropna(axis=1)
    x = np.linspace(0, 9570, num=320)
    x = x/22.5
    y = np.linspace(0, 7650, num=256)
    y = y/22.5
    
    df1 = df.iloc[100:136, 107:167]
    df1 = df1.reset_index(drop=True)
    df1.columns = range(df1.shape[1])
    x1 = np.linspace(143, 222, num=60)
    y1 = np.linspace(133, 181, num=36)    

    
    df1 = pd.DataFrame(df1)
    df2 = df1.iloc[:,30:42]
    # Find the maximum value in the DataFrame
    max_value = df2.values.max()
    max_values_left.append(max_value)
    # Find the column name where the maximum value belongs to
    mask = df1.isin([max_value])
    cols = mask.any()[mask.any()].index.tolist()
    col = cols[0]
    max_column_index.append(cols)
    
    count = df1.iloc[:, col]
    tck = interpolate.splrep(y1, count, s=2, k=4) 
    x_new = np.linspace(min(y1), max(y1), 1000)
    y_fit = interpolate.BSpline(*tck)(x_new)

    photon_dataframe_left[v] = y_fit
    peak_index = np.argmax(y_fit)
    peak_position = x_new[peak_index]
    verticle_peaks_left.append(peak_position)

for v in voltage:
    df = pd.read_csv("https://raw.githubusercontent.com/yd145763/OPA_data_analysis/main/"+v+"%20optimized%201092%20nm%202%20arms%20OPA.csv")
    df=df.dropna(axis=1)
    x = np.linspace(0, 9570, num=320)
    x = x/22.5
    y = np.linspace(0, 7650, num=256)
    y = y/22.5
    
    df1 = df.iloc[100:136, 107:167]
    df1 = df1.reset_index(drop=True)
    df1.columns = range(df1.shape[1])
    x1 = np.linspace(143, 222, num=60)
    y1 = np.linspace(133, 181, num=36)   
    
    X1,Y1 = np.meshgrid(x1,y1)
    colorbarmax = 5000
    fig = plt.figure(figsize=(8, 4))
    ax = plt.axes()
    cp=ax.contourf(X1,Y1,df1, 200, zdir='z', offset=-100, cmap='jet')
    clb=fig.colorbar(cp, ticks=(np.arange(0, colorbarmax, 500)).tolist())
    clb.ax.set_title('Photon/s', fontweight="bold")
    for l in clb.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(15)
    ax.set_xlabel('x-position (µm)', fontsize=18, fontweight="bold", labelpad=1)
    ax.set_ylabel('y-position (µm)', fontsize=18, fontweight="bold", labelpad=1)
    ax.xaxis.label.set_fontsize(18)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(18)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.show()
    plt.close()

ax2 = plt.axes()
columns = "2V", "4V", "6V", "8V", "10V"

for col in columns:
    count = photon_dataframe_left[col]    
    ax2.plot(x_new, count, label=col)
for col in columns[0:4]:
    count = photon_dataframe_left[col]  
    i = count[count == max(count)].index[0]
    ax2.scatter(x_new[i], count[i], marker = "x", s = 100)
count = photon_dataframe_left["10V"]
ax2.scatter(x_new[685], count[685], marker = "x", s = 100)   
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("y-position (µm)")
plt.ylabel("Photon/s")
plt.legend( prop={'weight': 'bold'})
plt.show()
plt.close()

ax2 = plt.axes()
columns = "12V", "14V", "16V", "17V"
for col in columns:
    count = photon_dataframe_left[col]    
    ax2.plot(x_new, count)
    ax2.tick_params(which='major', width=2.00)
    ax2.tick_params(which='minor', width=2.00)
    ax2.xaxis.label.set_fontsize(18)
    ax2.xaxis.label.set_weight("bold")
    ax2.yaxis.label.set_fontsize(18)
    ax2.yaxis.label.set_weight("bold")
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    plt.xlabel("y-position (µm)")
    plt.ylabel("Photon/s")
    plt.legend(["12V", "14V", "16V", "17V"], prop={'weight': 'bold'})
plt.show()
plt.close()

ax2 = plt.axes()
columns = "2V", "4V", "6V", "8V", "10V"
print(columns)
for col in columns:
    count = photon_dataframe_left[col]  
    i = count[count == max(count)].index[0]
    print(x_new[i])
for col in columns:
    count = photon_dataframe_right[col]    
    ax2.plot(x_new, count, label=col)
    i = count[count == max(count)].index[0]
    ax2.scatter(x_new[i], count[i], marker = "x", s = 100)
    ax2.tick_params(which='major', width=2.00)
    ax2.tick_params(which='minor', width=2.00)
    ax2.xaxis.label.set_fontsize(18)
    ax2.xaxis.label.set_weight("bold")
    ax2.yaxis.label.set_fontsize(18)
    ax2.yaxis.label.set_weight("bold")
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    plt.xlabel("y-position (µm)")
    plt.ylabel("Photon/s")
    plt.legend(prop={'weight': 'bold'})
plt.show()
plt.close()

ax2 = plt.axes()
columns = "12V", "14V", "16V", "17V"
for col in columns:
    count = photon_dataframe_right[col]    
    ax2.plot(x_new, count)
    ax2.tick_params(which='major', width=2.00)
    ax2.tick_params(which='minor', width=2.00)
    ax2.xaxis.label.set_fontsize(18)
    ax2.xaxis.label.set_weight("bold")
    ax2.yaxis.label.set_fontsize(18)
    ax2.yaxis.label.set_weight("bold")
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    plt.xlabel("y-position (µm)")
    plt.ylabel("Photon/s")
    plt.legend(["12V", "14V", "16V", "17V"], prop={'weight': 'bold'})
plt.show()
plt.close()

