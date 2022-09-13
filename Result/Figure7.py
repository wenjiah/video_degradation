# python run_answer_quality_avg.py --obj_name car --base_name jackson-town-square --test_date 2017-12-17 --ground_truth freq002_res608_yolo --ground_truth_freq 0.02 --predictions freq002_res608_yolo --sample_frac 0.5 --constraints freq002_res608_yolo --use_val False --val_frac 0.07 --our_alg True
# python run_answer_quality_avg.py --obj_name car --base_name jackson-town-square --test_date 2017-12-17 --ground_truth freq002_res608_yolo --ground_truth_freq 0.02 --predictions freq002_res608_yolo --sample_frac 0.5 --constraints freq002_res608_yolo --use_val True --val_frac 0.07 --our_alg True
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from Estimation.run_answer_quality_avg import estimate_avg_error

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fig = plt.figure(figsize=(7,3.5))

data_path = "../Data/"
obj_name = "car"
base_name = "night-street"
test_date = "2017-12-17"
ground_truth = "freq002_res608_yolo"
ground_truth_freq = 0.02
our_alg = "True"

#AVG; frame resolution
resolutions = [608, 416, 384, 320, 288, 256, 224, 192, 160, 128]
constraints = "freq002_res608_yolo"
cons_obj_name = None
val_frac = 0.07
sample_frac = 0.5
use_val = "False"

our_error = []
true_error = []
corrected_error = []

for resolution in resolutions:
    predictions = "freq002_res"+str(resolution)+"_yolo"
    true_err, estimate_err, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    our_error.append(estimate_err*100)
    true_error.append(true_err*100)

use_val = "True"
for resolution in resolutions:
    predictions = "freq002_res"+str(resolution)+"_yolo"
    _, estimate_err, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    corrected_error.append(estimate_err*100)

plt.plot(resolutions, our_error,linewidth=3, marker='x',markersize=7)
plt.plot(resolutions, corrected_error,linewidth=3, color = 'green', marker='x',markersize=7)
plt.plot(resolutions, true_error, linewidth=3, color='gold', marker='x',markersize=7)
plt.scatter(384, 35, s=2200, facecolors='none', linestyle='--', edgecolors='r')
plt.xlabel("Frame resolution",fontsize=18)
plt.ylabel("Analytical error (%)",fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.tight_layout()
fig.savefig("Figure7.pdf", bbox_inches='tight')