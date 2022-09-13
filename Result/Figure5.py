import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from Estimation.run_answer_quality_avg import estimate_avg_error
from Estimation.run_answer_quality_count import estimate_count_error

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fig = plt.figure(figsize=(9,3))

data_path = "../Data/"
obj_name = "car"
base_name = "UA-DETRAC"
test_date = "2020-09-19"
ground_truth = "freq1_res608_yolo"
predictions = "freq1_res608_yolo"
cons_obj_name = None
ground_truth_freq = 1
constraints = "freq1_res608_yolo"
use_val = "False"
val_frac = 1
our_alg = "CLT"
bound_check = True

#AVG
sample_fracs = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0012, 0.0014, 0.0016, 0.0018, 0.002, 0.0022, 0.0024, 0.0026, 0.0028, 0.003, 0.0032, 0.0034, 0.0036, 0.0038, 0.004]
CLT_fail_times = []

for sample_frac in sample_fracs:
    _, _, CLT_fail_time = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name, bound_check)
    CLT_fail_times.append(CLT_fail_time)


plt.subplot(1,2,1)
plt.plot(sample_fracs, CLT_fail_times,linewidth=3, marker='x',markersize=7)
plt.xlabel("Sample fraction",fontsize=16)
plt.ylabel("Percentage of failures(%)",fontsize=16)
plt.xticks([0.001, 0.002, 0.003, 0.004], fontsize=16)
plt.yticks(fontsize=16)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.ylim(0,15)
plt.tight_layout()
fig.text(0.25,-0.02,"(ⅰ) AVG",fontsize=16)

#COUNT
sample_fracs = [0.0002, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03]
CLT_fail_times = []

for sample_frac in sample_fracs:
    _, _, CLT_fail_time = estimate_count_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name, bound_check)
    CLT_fail_times.append(CLT_fail_time)


plt.subplot(1,2,2)
plt.plot(sample_fracs, CLT_fail_times,linewidth=3, marker='x',markersize=7)
plt.xlabel("Sample fraction",fontsize=16)
plt.ylabel("Percentage of failures(%)",fontsize=16)
plt.xticks([0.01, 0.02, 0.03], fontsize=16)
plt.yticks(fontsize=16)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.ylim(0,103)
plt.tight_layout()
fig.text(0.73,-0.02,"(ⅱ) COUNT",fontsize=16)
fig.savefig("Figure5.pdf", bbox_inches='tight')