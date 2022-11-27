import pickle,json,glob
import numpy as np
'''
files =  ['s9n_128_128_0_x.c', 's15_256_4096_add_0.c', 's15_64_512_add_0.c', 's6n_4096_mul_0.c', 's9n_16384_512_0_x.c', 's6_64_add_0.c', 's15_128_16384_mul_0.c', 's9n_2048_256_0_x.c', 's9n_16384_64_0_x.c', 's9n_1024_64_0_x.c', 's15_64_2048_sub_0.c', 's15_2048_2048_mul_0.c', 's15_256_512_add_0.c', 's15_64_4096_add_0.c', 's8n_4096_sub_0.c', 's15_512_512_add_0.c', 's15_4096_1024_mul_0.c', 's6_128_sub_0.c', 's15_128_2048_add_0.c', 's8n_128_add_0.c', 's12nn_64_2_2_sa_ia.c', 's9n_8192_256_0_x.c', 's15_8192_8192_sub_0.c', 's12nn_512_2_2_sa_ia.c', 's9n_2048_64_0_x.c']


f = open('runtimes.pickle', 'rb')
runtimes = pickle.load(f)    
f.close()


emb = {}
with open('embeddings_8.json') as f:
    emb = json.load(f)

vf_if = {}
files_VF_IF = runtimes.keys()
vf_list = []
if_list = []
for file_VF_IF in files_VF_IF:
    tmp = file_VF_IF.split('.')
    #print(tmp)
    fn = tmp[0]
    tmp = tmp[1].split('-')
    VF = int(tmp[0])
    IF = int(tmp[1])
    vf_list.append(VF)
    if_list.append(IF)
    rt_mean = np.mean(runtimes[file_VF_IF])
    if fn not in vf_if.keys():
        vf_if[fn] = (rt_mean, VF, IF)
    else:
        rt_mean_pre = vf_if[fn][0]
        if rt_mean < rt_mean_pre:
            vf_if[fn] = (rt_mean, VF, IF)

for file_VF_IF in files_VF_IF:
    tmp = file_VF_IF.split('.')
    #print(tmp)
    fn = tmp[0]
    if (fn + '.c' in files):
        print("vf_if = ", vf_if[fn])
'''

def insert_pragma(save_dir, run_dir, fname, VF, IF, insert_timer):
    file = fname.split('.')[0]
    new_file = save_dir + '/' + file + ".c"
    fw = open(new_file, 'w')
    with open(run_dir+'/'+fname) as fr:
        for line in fr:
            #print(line.rstrip())
            if re.match(r'^\s*for\s*\(',line) or re.match(r'^\s*while\s*\(',line):
                fw.write("#pragma clang loop vectorize_width("+str(VF)+") interleave_count("+str(IF)+")\n")         
            if (insert_timer):
                if (re.match(r'\s+return 0;', line)):
                    fw.write("gettimeofday(&end, NULL);\n") 
                    fw.write("double time_taken = end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6;\n")
                    fw.write("printf(\"time program took %f seconds to execute\\n\", time_taken);\n")
            fw.write(line)
            if (insert_timer):
                if (re.match(r'int\s*main\(', line)): 
                    fw.write("struct timeval start, end;\ngettimeofday(&start, NULL);\n") 
    fw.close()

def measure_execution_time(save_dir, run_dir, file_name, VF, IF, repeat):

    insert_pragma(save_dir, run_dir, file_name, VF, IF, 1)

    os.system("clang -O3 " + save_dir+'/'+file_name + ' ' + run_dir+'/header.c')
    time_slots = []
    for i in range(repeat):
        output = subprocess.check_output("./a.out", shell=True)
        time = float(output.split()[3])
        time_slots.append(time)

    average = sum(time_slots) / len(time_slots)

    return (average, time_slots)

f = open('runtimes.pickle', 'rb')
runtimes = pickle.load(f)    
f.close()

run_dir = "training_data_default"
save_dir = "training_data_vec"
vf_list = [1,2,4,8,16,32,64] # TODO: change this to match your hardware
if_list = [1,2,4,8,16] # TODO: change this to match your hardware
runtimes1 = {}
runtimes2 = {}
for file in glob.glob(run_dir+"/*.c"):
    print(file)
    file_name = file.split('/')[1]
    if (file_name == "header.c"):
        continue
    fn = file_name.split('.')[0]
    avg, times = measure_execution_time(save_dir, run_dir, file_name, VF, IF, 10)
