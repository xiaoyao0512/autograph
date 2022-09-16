import pickle
import numpy as np

f = open('runtimes_icx7_omp_orig.pickle', 'rb')
base_runtimes = pickle.load(f)
f.close()
	
f = open('runtimes_omp_icx_8classes.pickle', 'rb')
runtimes = pickle.load(f)
f.close()

vf_if = {}
times = {}
files_VF = runtimes.keys()
vf_list = []
if_list = []
baseline = {}
labels = {}

for file_VF in files_VF:
	times[file_VF] = {}
	kernel_runtimes = runtimes[file_VF]
	for k, v in kernel_runtimes.items():
		kernel_runtimes[k] = np.mean(v)
		times[file_VF][k] = np.mean(v)
	labels[file_VF] = min(kernel_runtimes, key=kernel_runtimes.get)
	rt_mean = min(kernel_runtimes.values())
	base_mean = np.mean(base_runtimes[file_VF])
	vf_if[file_VF] = (rt_mean, labels[file_VF])
	baseline[file_VF] = base_mean


print(vf_if["ALPBench_v1.0/csuFaceIdEval_5.1/csuEBGMMeasure.c_main_line307_0"])

print(vf_if["FreeBench_default/pcompress2/arithmetic.c_start_model_line188_24"])
