import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from Estimation.run_answer_quality_avg import estimate_avg_error
from Estimation.run_answer_quality_max import estimate_max_error

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fig = plt.figure(figsize=(9,8))

data_path = "../Data/"
obj_name = "car"
base_name = "night-street"
test_date = "2017-12-17"
ground_truth = "freq002_res640_mrcnn"
ground_truth_freq = 0.02
our_alg = "True"

#AVG; sample fraction
predictions = "freq002_res640_mrcnn"
constraints = "freq002_res608_yolo"
cons_obj_name = None
val_frac = 0.06
sample_fracs = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
use_val = "False"

our_error = []
true_error = []
corrected_error = []

for sample_frac in sample_fracs:
    true_err, estimate_err, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    our_error.append(estimate_err*100)
    true_error.append(true_err*100)

use_val = "True"
for sample_frac in sample_fracs:
    _, estimate_err, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    corrected_error.append(estimate_err*100)

plt.subplot(3,2,1)
plt.plot(sample_fracs, our_error,linewidth=3, marker='x',markersize=7)
plt.plot(sample_fracs, corrected_error,linewidth=3, color = 'green', marker='x',markersize=7)
plt.plot(sample_fracs, true_error,linewidth=3, color='gold', marker='x',markersize=7)
plt.xlabel("Sample fraction",fontsize=16)
plt.ylabel("Analytical error (%)",fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
fig.legend(["Error bound estimation w/o correction set", "Error bound estimation w/ correction set", "True error"],bbox_to_anchor=(0.5,1.11), loc='upper center',ncol=2,fontsize=16)
plt.ylim(-3,80)
plt.tight_layout()

#MAX; sample fraction
predictions = "freq002_res640_mrcnn"
constraints = "freq002_res608_yolo"
cons_obj_name = None
val_frac = 0.02
sample_fracs = [0.00025, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.001, 0.0015, 0.002, 0.003]
use_val = "False"

our_error = []
true_error = []
corrected_error = []

for sample_frac in sample_fracs:
    true_err, estimate_err, _ = estimate_max_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    our_error.append(estimate_err*100)
    true_error.append(true_err*100)

use_val = "True"
for sample_frac in sample_fracs:
    _, estimate_err, _ = estimate_max_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    corrected_error.append(estimate_err*100)

plt.subplot(3,2,2)
plt.plot(sample_fracs, our_error,linewidth=3, marker='x',markersize=7)
plt.plot(sample_fracs, corrected_error,linewidth=3, color = 'green', marker='x',markersize=7)
plt.plot(sample_fracs, true_error, linewidth=3, color='gold', marker='x',markersize=7)
plt.xlabel("Sample fraction",fontsize=16)
plt.ylabel("Analytical error (%)",fontsize=16)
xticks = [0.001, 0.002, 0.003]
plt.xticks(xticks, fontsize=15)
plt.yticks(fontsize=15)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.ylim(-3,80)
plt.tight_layout()

#AVG; frame resolution
resolutions = [640, 576, 512, 448, 384, 320, 256, 192]
constraints = "freq002_res608_yolo"
cons_obj_name = None
val_frac = 0.06
sample_frac = 0.5
use_val = "False"

our_error = []
true_error = []
corrected_error = []

for resolution in resolutions:
    predictions = "freq002_res"+str(resolution)+"_mrcnn"
    true_err, estimate_err, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    our_error.append(estimate_err*100)
    true_error.append(true_err*100)

use_val = "True"
for resolution in resolutions:
    predictions = "freq002_res"+str(resolution)+"_mrcnn"
    _, estimate_err, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    corrected_error.append(estimate_err*100)

plt.subplot(3,2,3)
plt.plot(resolutions, our_error,linewidth=3, marker='x',markersize=7)
plt.plot(resolutions, corrected_error,linewidth=3, color = 'green', marker='x',markersize=7)
plt.plot(resolutions, true_error, linewidth=3, color='gold', marker='x',markersize=7)
plt.xlabel("Frame resolution",fontsize=16)
plt.ylabel("Analytical error (%)",fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ellipse = matplotlib.patches.Ellipse(xy = (410,18), width = 460, height=25, angle=-3, facecolor = 'none', edgecolor = 'r', linestyle = '--')
ax.add_patch(ellipse)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.tight_layout()

#MAX; frame resolution
resolutions = [640, 576, 512, 448, 384, 320, 256, 192]
constraints = "freq002_res608_yolo"
cons_obj_name = None
val_frac = 0.02
sample_frac = 0.5
use_val = "False"

our_error = []
true_error = []
corrected_error = []

for resolution in resolutions:
    predictions = "freq002_res"+str(resolution)+"_mrcnn"
    true_err, estimate_err, _ = estimate_max_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    our_error.append(estimate_err*100)
    true_error.append(true_err*100)

use_val = "True"
for resolution in resolutions:
    predictions = "freq002_res"+str(resolution)+"_mrcnn"
    _, estimate_err, _ = estimate_max_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    corrected_error.append(estimate_err*100)

plt.subplot(3,2,4)
plt.plot(resolutions, our_error,linewidth=3, marker='x',markersize=7)
plt.plot(resolutions, corrected_error,linewidth=3, color = 'green', marker='x',markersize=7)
plt.plot(resolutions, true_error, linewidth=3, color='gold', marker='x',markersize=7)
plt.xlabel("Frame resolution",fontsize=16)
plt.ylabel("Analytical error (%)",fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ellipse = matplotlib.patches.Ellipse(xy = (280,5), width = 200, height=12, angle=0, facecolor = 'none', edgecolor = 'r', linestyle = '--')
ax.add_patch(ellipse)
plt.ylim(-5,140)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.tight_layout()

#AVG; restricted class
predictions = "freq002_res640_mrcnn"
val_frac = 0.06
sample_frac = 0.5
use_val = "False"
constraints_list = ["freq002_res608_yolo", "freq002_face_mtcnn", "freq002_res608_yolo"]
cons_obj_name_list = [None, "face", "person"]

our_error = []
true_error = []
corrected_error = []

for i in range(3):
    constraints = constraints_list[i]
    cons_obj_name = cons_obj_name_list[i]
    true_err, estimate_err, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    our_error.append(estimate_err*100)
    true_error.append(true_err*100)

use_val = "True"
for i in range(3):
    constraints = constraints_list[i]
    cons_obj_name = cons_obj_name_list[i]
    _, estimate_err, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    corrected_error.append(estimate_err*100)

plt.subplot(3,2,5)
label_list = ["No restriction   ", "Face", "Person"]
x = [0,1.2,2.4]
plt.bar(x=x, height=our_error, width=0.25, zorder=2)
plt.bar(x=[i+0.25 for i in x], height=corrected_error, width=0.25, color='green', zorder=2)
plt.bar(x=[i+0.5 for i in x], height=true_error, width=0.25, color='gold', zorder=2)
plt.xlabel("Restricted class",fontsize=16)
plt.ylabel("Analytical error (%)",fontsize=16)
plt.xticks([i + 0.25 for i in x], label_list, fontsize=15)
yticks = [0,10,20,30]
plt.yticks(yticks,fontsize=15)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ellipse = matplotlib.patches.Ellipse(xy = (2.4,3), width = 0.4, height=9.5, angle=0, facecolor = 'none', edgecolor = 'r', linestyle = '--')
ax.add_patch(ellipse)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.tight_layout()

#MAX; restricted class
predictions = "freq002_res640_mrcnn"
val_frac = 0.02
sample_frac = 0.5
use_val = "False"
constraints_list = ["freq002_res608_yolo", "freq002_face_mtcnn", "freq002_res608_yolo"]
cons_obj_name_list = [None, "face", "person"]

our_error = []
true_error = []
corrected_error = []

for i in range(3):
    constraints = constraints_list[i]
    cons_obj_name = cons_obj_name_list[i]
    true_err, estimate_err, _ = estimate_max_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    our_error.append(estimate_err*100)
    true_error.append(true_err*100)

use_val = "True"
for i in range(3):
    constraints = constraints_list[i]
    cons_obj_name = cons_obj_name_list[i]
    _, estimate_err, _ = estimate_max_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    corrected_error.append(estimate_err*100)

plt.subplot(3,2,6)
label_list = ["No restriction   ", "Face", "Person"]
x = [0,1.2,2.4]
plt.bar(x=x, height=our_error, width=0.25, zorder=2)
plt.bar(x=[i+0.25 for i in x], height=corrected_error, width=0.25, color='green', zorder=2)
plt.bar(x=[i+0.5 for i in x], height=true_error, width=0.25, color='gold', zorder=2)
plt.xlabel("Restricted class",fontsize=16)
plt.ylabel("Analytical error (%)",fontsize=16)
plt.xticks([i + 0.25 for i in x], label_list, fontsize=15)
yticks = [0,10,20,30]
plt.yticks(yticks,fontsize=15)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.ylim(0,33)
plt.tight_layout()
fig.text(0.25,0,"(ⅰ) AVG",fontsize=16)
fig.text(0.75,0,"(ⅱ) MAX",fontsize=16)
fig.savefig("Figure6a.pdf", bbox_inches='tight')