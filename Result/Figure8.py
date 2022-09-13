import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter

import sys
sys.path.append('../')

from Analyze.car_distribution import car_num

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fig = plt.figure(figsize=(12,3.5))
matplotlib.rcParams.update({'font.size': 13})

data_path = "../Data/"
obj_name = "car"
base_name = "night-street"
test_date = "2017-12-17"
ground_truth_freq = 0.02

#608x608
predictions = "freq002_res608_yolo"
car_num_list = car_num(data_path, obj_name, base_name, test_date, predictions, ground_truth_freq)


plt.subplot(1,3,1)
x = [1,2,3,4,5]
label_list = ["0", "1", "2", "3", ">3"]
plt.bar(x=[i for i in x], height=car_num_list, width=0.6, zorder=2)
plt.xticks([i for i in x], label_list,fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Predicted car number", fontsize=18)
plt.ylabel("# of frames", fontsize=18)
plt.ylim((0,12000))
plt.title("  Res: 608x608", fontsize = 18)
ax=plt.gca()
xfmt = ScalarFormatter(useMathText=True)
xfmt.set_powerlimits((0, 0)) 
ax.yaxis.set_major_formatter(xfmt)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.tight_layout()

#384x384
predictions = "freq002_res384_yolo"
car_num_list = car_num(data_path, obj_name, base_name, test_date, predictions, ground_truth_freq)


plt.subplot(1,3,2)
x = [1,2,3,4,5]
label_list = ["0", "1", "2", "3", ">3"]
plt.bar(x=[i for i in x], height=car_num_list, width=0.6, zorder=2)
plt.xticks([i for i in x], label_list,fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Predicted car number", fontsize=18)
plt.ylabel("# of frames", fontsize=18)
plt.ylim((0,12000))
plt.title("  Res: 384x384", fontsize = 18)
ax=plt.gca()
xfmt = ScalarFormatter(useMathText=True)
xfmt.set_powerlimits((0, 0))  # Or whatever your limits are . . .
ax.yaxis.set_major_formatter(xfmt)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.tight_layout()

#320x320
predictions = "freq002_res320_yolo"
car_num_list = car_num(data_path, obj_name, base_name, test_date, predictions, ground_truth_freq)


plt.subplot(1,3,3)
x = [1,2,3,4,5]
label_list = ["0", "1", "2", "3", ">3"]
plt.bar(x=[i for i in x], height=car_num_list, width=0.6, zorder=2)
plt.xticks([i for i in x], label_list,fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Predicted car number", fontsize=18)
plt.ylabel("# of frames", fontsize=18)
plt.ylim((0,12000))
plt.title("  Res: 320x320", fontsize = 18)
ax=plt.gca()
xfmt = ScalarFormatter(useMathText=True)
xfmt.set_powerlimits((0, 0))  # Or whatever your limits are . . .
ax.yaxis.set_major_formatter(xfmt)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.tight_layout()
fig.savefig("Figure8.pdf", bbox_inches='tight')