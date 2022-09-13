import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from Estimation.run_answer_quality_avg import estimate_avg_error
from Estimation.run_answer_quality_sum import estimate_sum_error
from Estimation.run_answer_quality_count import estimate_count_error
from Estimation.run_answer_quality_max import estimate_max_error

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fig = plt.figure(figsize=(19,3.5))

data_path = "../Data/"
obj_name = "car"
base_name = "night-street"
test_date = "2017-12-17"
ground_truth = "freq002_res640_mrcnn"
predictions = "freq002_res640_mrcnn"
cons_obj_name = None
ground_truth_freq = 0.02
constraints = "freq002_res608_yolo"
use_val = "False"
val_frac = 1

# AVG
our_algs = ["True", "EBGS", "HoeffdingSerfling", "Hoeffding", "CLT"]
sample_fracs = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1]
estimate_err_list = []
true_err_list = []

for our_alg in our_algs:
    estimate_errs = []
    true_errs = []
    for sample_frac in sample_fracs:
        true_err, estimate_err, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
        estimate_errs.append(estimate_err*100)
        true_errs.append(true_err*100)
    estimate_err_list.append(estimate_errs)
    true_err_list.append(true_errs)


plt.subplot(1,4,1)
plt.plot(sample_fracs, estimate_err_list[0],linewidth=3)
plt.plot(sample_fracs, true_err_list[0], linestyle='--',linewidth=3, color='deepskyblue')
plt.plot(sample_fracs, estimate_err_list[1],linewidth=3, color = 'darkgreen')
plt.plot(sample_fracs, true_err_list[1], linestyle='--',linewidth=3, color='limegreen')
plt.plot(sample_fracs, estimate_err_list[2],linewidth=3, color = 'pink')
plt.plot(sample_fracs, true_err_list[2], linestyle='--',linewidth=3, color='lightpink')
plt.plot(sample_fracs, estimate_err_list[3],linewidth=3, color = 'orange')
plt.plot(sample_fracs, true_err_list[3], linestyle='--',linewidth=3, color='gold')
plt.plot(sample_fracs, estimate_err_list[4],linewidth=3, color = 'darkviolet')
plt.plot(sample_fracs, true_err_list[4], linestyle='--',linewidth=3, color='violet')
plt.plot(sample_fracs, [-5]*12, linewidth=3, color='brown')
plt.plot(sample_fracs, [-5]*12, linestyle='--',linewidth=3, color='chocolate')
plt.xlabel("Sample fraction",fontsize=18)
plt.ylabel("Analytical error (%)",fontsize=18)
plt.xticks(fontsize=16)
plt.yticks([0, 50, 100, 150], fontsize=16)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.ylim(-3,150)
fig.legend(["Upper bound from Smokescreen                    ", "True error of Smokescreen", "Upper bound from EBGS", "True error of EBGS", "Upper bound from Hoeffding-Serfling                    ", "True error of Hoeffding-Serfling", "Upper bound from Hoeffding", "True error of Hoeffding", "Upper bound from CLT", "True error of CLT", "Upper bound from Stein", "True error of Stein"],bbox_to_anchor=(0.5,1.36), loc='upper center',ncol=3,fontsize=15)
plt.tight_layout()
fig.text(0.12,-0.02,"(ⅰ) AVG",fontsize=18)

#SUM
our_algs = ["True", "EBGS", "HoeffdingSerfling", "Hoeffding", "CLT"]
sample_fracs = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1]
estimate_err_list = []
true_err_list = []

for our_alg in our_algs:
    estimate_errs = []
    true_errs = []
    for sample_frac in sample_fracs:
        true_err, estimate_err, _ = estimate_sum_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
        estimate_errs.append(estimate_err*100)
        true_errs.append(true_err*100)
    estimate_err_list.append(estimate_errs)
    true_err_list.append(true_errs)


plt.subplot(1,4,2)
plt.plot(sample_fracs, estimate_err_list[0],linewidth=3)
plt.plot(sample_fracs, true_err_list[0], linestyle='--',linewidth=3, color='deepskyblue')
plt.plot(sample_fracs, estimate_err_list[1],linewidth=3, color = 'darkgreen')
plt.plot(sample_fracs, true_err_list[1], linestyle='--',linewidth=3, color='limegreen')
plt.plot(sample_fracs, estimate_err_list[2],linewidth=3, color = 'pink')
plt.plot(sample_fracs, true_err_list[2], linestyle='--',linewidth=3, color='lightpink')
plt.plot(sample_fracs, estimate_err_list[3],linewidth=3, color = 'orange')
plt.plot(sample_fracs, true_err_list[3], linestyle='--',linewidth=3, color='gold')
plt.plot(sample_fracs, estimate_err_list[4],linewidth=3, color = 'darkviolet')
plt.plot(sample_fracs, true_err_list[4], linestyle='--',linewidth=3, color='violet')
plt.xlabel("Sample fraction",fontsize=18)
plt.ylabel("Analytical error (%)",fontsize=18)
plt.xticks(fontsize=16)
plt.yticks([0,50,100,150,200,250], fontsize=16)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.ylim(-3,150)
plt.tight_layout()
fig.text(0.36,-0.02,"(ⅱ) SUM",fontsize=18)

#COUNT
our_algs = ["True", "EBGS", "HoeffdingSerfling", "Hoeffding", "CLT"]
sample_fracs = [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.006, 0.007, 0.008, 0.01, 0.02, 0.05]
estimate_err_list = []
true_err_list = []

for our_alg in our_algs:
    estimate_errs = []
    true_errs = []
    for sample_frac in sample_fracs:
        true_err, estimate_err, _ = estimate_count_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
        estimate_errs.append(estimate_err*100)
        true_errs.append(true_err*100)
    estimate_err_list.append(estimate_errs)
    true_err_list.append(true_errs)


plt.subplot(1,4,3)
plt.plot(sample_fracs, estimate_err_list[0],linewidth=3)
plt.plot(sample_fracs, true_err_list[0], linestyle='--',linewidth=3, color='deepskyblue')
plt.plot(sample_fracs, estimate_err_list[1],linewidth=3, color = 'darkgreen')
plt.plot(sample_fracs, true_err_list[1], linestyle='--',linewidth=3, color='limegreen')
plt.plot(sample_fracs, estimate_err_list[2],linewidth=3, color = 'pink')
plt.plot(sample_fracs, true_err_list[2], linestyle='--',linewidth=3, color='lightpink')
plt.plot(sample_fracs, estimate_err_list[3],linewidth=3, color = 'orange')
plt.plot(sample_fracs, true_err_list[3], linestyle='--',linewidth=3, color='gold')
plt.plot(sample_fracs, estimate_err_list[4],linewidth=3, color = 'darkviolet')
plt.plot(sample_fracs, true_err_list[4], linestyle='--',linewidth=3, color='violet')
plt.xlabel("Sample fraction",fontsize=18)
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
fig.text(0.6,-0.02,"(ⅲ) COUNT",fontsize=18)

#MAX
our_algs = ["True", "Stein"]
sample_fracs = [0.00025, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.001, 0.0015]
estimate_err_list = []
true_err_list = []

for our_alg in our_algs:
    estimate_errs = []
    true_errs = []
    for sample_frac in sample_fracs:
        true_err, estimate_err, _ = estimate_max_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
        estimate_errs.append(estimate_err*100)
        true_errs.append(true_err*100)
    estimate_err_list.append(estimate_errs)
    true_err_list.append(true_errs)


plt.subplot(1,4,4)
plt.plot(sample_fracs, estimate_err_list[0],linewidth=3)
plt.plot(sample_fracs, true_err_list[0], linestyle='--',linewidth=3, color='deepskyblue')
plt.plot(sample_fracs, estimate_err_list[1], linewidth=3, color='brown')
plt.plot(sample_fracs, true_err_list[1], linestyle='--',linewidth=3, color='chocolate')
plt.xlabel("Sample fraction",fontsize=18)
plt.ylabel("Analytical error (%)",fontsize=18)
xticks = [0.0005, 0.001, 0.0015]
plt.xticks(xticks, fontsize=16)
plt.yticks(fontsize=16)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.ylim(-3,105)
plt.tight_layout()
fig.text(0.86,-0.02,"(ⅳ) MAX",fontsize=18)
fig.savefig("Figure4a.pdf", bbox_inches='tight')