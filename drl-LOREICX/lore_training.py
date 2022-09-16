'''
Copyright (c) 2019, Ameer Haj Ali (UC Berkeley), and Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from envs.lore_graphsage import NeuroVectorizerEnv
from ray.tune.registry import register_env
from ray.tune.logger import TBXLogger
import argparse
import json, pickle
import torch
import numpy as np

from ray.rllib.agents.registry import get_trainer_class
from random import randint 

import glob

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument("--num-cpus", type=int, default=2)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument("--eager-tracing", action="store_true")


parser.add_argument(
    "--explore-during-inference",
    action="store_true",
    help="Whether the trained policy should use exploration during action "
    "inference.",
)


register_env("autovec", lambda config:NeuroVectorizerEnv(config))



if __name__ == "__main__":
    ray.init()
    args = parser.parse_args()
    #timesteps = [1000, 5000, 10000, 50000, 100000, 200000, 300000, 400000]
    timestep = 500000
    reward_mean_list = []
    reward_min_list = []
    reward_max_list = []
    accuracy_list = []
    total_loss_list = []
    exec2_list = []
    exec3_list = []
    exec4_list = []
    exec5_list = []
    predictions = {}
    train_batch_size_space = [1000, 5000, 10000, 15000]
    lr_space = [1e-5, 1e-4, 1e-3]
    minibatch_space = [100, 500, 1000]
    max_speedup_O3 = float('-inf')
    max_param = (0, 0, 0)
    for batch_size in train_batch_size_space:
        for lr in lr_space:
            for minibatch in minibatch_space:
                env_config = {'dirpath':'./json_lore','new_rundir':'./new_garbage'}
                config = {
                            #"sample_batch_size": 25,
                            "train_batch_size": batch_size,
                            "sgd_minibatch_size": minibatch,
                            "num_sgd_iter": minibatch,
                            "lr":lr,
                            #"vf_loss_coeff":0.5,
                            "env": "autovec",
                            "horizon":  1,
                            "num_gpus": 1,
                            "model":{'fcnet_hiddens':[256, 256]},
                            "num_workers": 4,
                            "env_config":env_config,
                            "framework": args.framework,
                            # Run with tracing enabled for tfe/tf2?
                            "eager_tracing": args.eager_tracing,
                            }
                results = tune.run("PPO",
                        #restore = "~/ray_results/PPO_*/checkpoint_240/checkpoint-240",
                        checkpoint_freq  = 10,
                        checkpoint_at_end=True,
                        name = "neurovectorizer_train",
                        stop = {
                            "timesteps_total": timestep,
                        },
                        config=config,
                        loggers=[TBXLogger]
                )

                
                print("Training completed. Restoring new Trainer for action inference.")
                # Get the last checkpoint from the above training run.
                checkpoint = results.get_last_checkpoint()
                # Create new Trainer and restore its state from the last checkpoint.
                trainer = get_trainer_class(args.run)(config=config)
                trainer.restore(checkpoint)

                # Create the env to do inference in.
                env = NeuroVectorizerEnv(env_config)
                # obs = env.reset()
                #print(env.new_testfiles)
                root = './'+env.new_testfiles[0].split('/')[1]+'/'
                num_episodes = 0
                episode_reward = 0.0



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

                #rewards = [] 
                acc = 0
                exec1 = 0
                exec2 = 0
                exec3 = 0
                exec4 = 0
                exec5 = 0
                cnt = 0
                VF_list = [0,1,2,3,4,5,6] # TODO: change this to match your hardware
                typ = 'max'
                with open('lore_omp_embeddings2_graphsage_gcn256_icx8_without_nopragma.json') as f:
                    features = json.load(f)
                feats = features
                # labels = features["labels"]
                files = list(features.keys())
                files_root = []

                with open('lore_testing_subset_spec2006_icx7_omp.json') as f:
                    files_temp= json.load(f)
                files_test = []
                #to deal with the error cases when kernels in the given test set are missing in the runtimes file or embeddings file
                for file in files_temp:
                    if file in features.keys() and file in labels.keys():
                        files_test.append(file)
                        
                env.new_testfiles = files_test

                rewards = [] 
                acc = 0
                predictions = {}
                for fn in files_test:
                    label = labels[fn]
                    feat = feats[fn]
                    obs = torch.FloatTensor(feat).numpy()
                    print("f = ", fn)
                    # Compute an action (`a`).
                    a = trainer.compute_single_action(
                        observation=obs,
                        explore=args.explore_during_inference,
                        policy_id="default_policy",  # <- default value
                    )
                    predictions[fn] = int(a)
                    print("label = ", label, a)
                    if (label == a):
                        acc += 1
                    VF_pred = a
                    VF_rand = randint(0, 6)
                    t1 = times[fn][label]
                    t2 = times[fn][VF_pred]
                    t3 = baseline[fn]
                    t4 = times[fn][VF_rand]
                    print("t1 = ", t1, ", t2 = ", t2, ", t3 = ", t3, ", t4 = ", t4)
                    exec1 += abs(t1 - t2)
                    speedup_gt = ((t1 - t2) / t1)
                    speedup_base = ((t3 - t2) / t3)
                    exec2 += speedup_gt
                    speedup_gt_rand = ((t1 - t4) / t1)
                    speedup_base_rand = ((t3 - t4) / t3)
                    print("speedup compared to ground truth = ", speedup_gt)
                    exec3 += speedup_base
                    print("reward = ", speedup_base)
                    print("speedup compared to baseline O3 = ", speedup_base)
                    #if ((abs(speedup_gt) > 2) or (abs(speedup_base) > 2)):
                    #    bad.append(fn)
                    exec4 += speedup_gt_rand
                    exec5 += speedup_base_rand

                acc = acc / len(files_test) * 100.0
                exec1 = exec1 / len(files_test)
                exec2 = exec2 / len(files_test) * 100
                exec3 = exec3 / len(files_test) * 100
                exec4 = exec4 / len(files_test) * 100
                exec5 = exec5 / len(files_test) * 100

                ID = list(results.results.keys())[0]
                reward_mean_list.append(results.results[ID]['episode_reward_mean'])
                reward_min_list.append(results.results[ID]['episode_reward_min'])
                reward_max_list.append(results.results[ID]['episode_reward_max'])
                accuracy_list.append(acc)
                total_loss_list.append(results.results[ID]['info']['learner']['default_policy']['learner_stats']['total_loss'])
                exec2_list.append(exec2)
                exec3_list.append(exec3)
                exec4_list.append(exec4)
                exec5_list.append(exec5)
                if exec3 > max_speedup_O3:
                    max_speedup_O3 = exec3
                    max_param = (batch_size, lr, minibatch)


    #print("results = ", results)
    print("max_speedup_O3 = ", max_speedup_O3)
    print("max_param = ", max_param)
    ray.shutdown()
