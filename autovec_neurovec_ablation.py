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
from envs.neurovec_test import NeuroVectorizerEnv
from ray.tune.registry import register_env
from ray.tune.logger import TBXLogger
import argparse
import json, pickle
import torch
import numpy as np

from ray.rllib.agents.registry import get_trainer_class
from random import randint 

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument("--num-cpus", type=int, default=0)
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
    timesteps = [1000, 5000, 10000, 50000, 100000, 200000, 300000, 400000, 500000]
    #timesteps = [500000]
    reward_mean_list = []
    reward_min_list = []
    reward_max_list = []
    accuracy_list = []
    total_loss_list = []
    exec2_list = []
    exec3_list = []
    exec4_list = []
    exec5_list = []
    for timestep in timesteps:
        env_config = {'dirpath':'./training_data_default','new_rundir':'./new_garbage'}
        config = {
                    #"sample_batch_size": 25,
                    "train_batch_size": 500,
                    "sgd_minibatch_size": 20,
                    "num_sgd_iter":20,
                    #"lr":5e-5,
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
        obs = env.reset()

        num_episodes = 0
        episode_reward = 0.0


        f = open('runtimes.pickle', 'rb')
        runtimes = pickle.load(f)    
        f.close()
        f = open('runtimes_none_pragma.pickle', 'rb')
        base_runtimes = pickle.load(f)    
        f.close()

        vf_if = {}
        times = {}
        files_VF_IF = runtimes.keys()
        vf_list = []
        if_list = []
        baseline = {}
        for file_VF_IF in files_VF_IF:
            tmp = file_VF_IF.split('.')
            fn = tmp[0]
            tmp = tmp[1].split('-')
            VF = int(tmp[0])
            IF = int(tmp[1])
            fn_c = fn + '.c'
            rt_mean = np.median(runtimes[file_VF_IF])
            base_mean = np.median(base_runtimes[fn])
            #print("filename = ", fn)
            #print("VF = ", VF)
            #print("IF = ", IF)
            #print("mean = ", rt_mean)
            if fn_c not in vf_if.keys():
                vf_if[fn_c] = (rt_mean, VF, IF)
            else:
                rt_mean_pre = vf_if[fn_c][0]
                if rt_mean < rt_mean_pre:
                    vf_if[fn_c] = (rt_mean, VF, IF)    
            if fn_c not in times.keys():
                times[fn_c] = {}
            if VF not in times[fn_c].keys():
                times[fn_c][VF] = {}
            if IF not in times[fn_c][VF].keys():
                times[fn_c][VF][IF] = rt_mean
            else:
                rt_mean_pre = times[fn][VF][IF]
                if (rt_mean < rt_mean_pre):
                    times[fn_c][VF][IF] = rt_mean

            if fn_c not in baseline.keys():
                baseline[fn_c] = base_mean
            else:
                base_mean_pre = baseline[fn_c]
                if base_mean < base_mean_pre:
                    baseline[fn_c] = base_mean
        #while num_episodes < args.num_episodes_during_inference:
        # iterate the dataset



        
        #rewards = [] 
        acc = 0
        exec1 = 0
        exec2 = 0
        exec3 = 0
        exec4 = 0
        exec5 = 0
        cnt = 0
        VF_list = [1,2,4,8,16] # TODO: change this to match your hardware
        IF_list = [1,2,4,8,16] # TODO: change this to match your hardware
        dim = 64
        typ = 'max'
        with open('features.json') as f:
            features = json.load(f)
        feats = features["feat"]
        labels = features["labels"]
        files = features["files"] 
        rewards = [] 
        acc = 0
        for fidx in range(len(files)):
            f = files[fidx] 
            label = labels[fidx]
            feat = feats[fidx][0]
            #obs = torch.FloatTensor(feat).numpy()
            obs = []  
            #print("f = ", f)      
            #print("feat = ", feat)
            #print("len = ", len(feat))
            obs.append(torch.IntTensor(feat[0:200]).numpy())
            obs.append(torch.IntTensor(feat[200:400]).numpy())
            obs.append(torch.IntTensor(feat[400:600]).numpy())
            obs.append(torch.FloatTensor(feat[600:800]).numpy())
            #print("obs = ", obs)
            # Compute an action (`a`).
            #print("obs = ", obs)
            # Compute an action (`a`).
            a = trainer.compute_single_action(
                observation=obs,
                explore=args.explore_during_inference,
                policy_id="default_policy",  # <- default value
            )
            #print("action = ", a)
            #print("label = ", label)
            y1 = VF_list[int(int(label) / 5)]
            y2 = IF_list[int(int(label) % 5)]
            #print("label = ", label, a)
            if (y1 == VF_list[a[0]] and y2 == IF_list[a[1]]):
                acc += 1
            # Send the computed action `a` to the env.
            _, reward, _, _ = env.step(a)
            #print("reward = ", reward)
            #rewards.append(reward)
            VF_pred = VF_list[a[0]]
            IF_pred = IF_list[a[1]]
            VF_rand = VF_list[randint(0, 4)]
            IF_rand = IF_list[randint(0, 4)]
            #print(VF_pred, IF_pred)
            #print(sampled_y1, sampled_y2)
            t1 = times[f][y1][y2]
            t2 = times[f][VF_pred][IF_pred]
            t3 = baseline[f]
            t4 = times[f][VF_rand][IF_rand]
            #print("t1 = ", t1, ", t2 = ", t2, ", t3 = ", t3, ", t4 = ", t4)
            exec1 += abs(t1 - t2)
            speedup_gt = ((t1 - t2) / t1)
            speedup_base = ((t3 - t2) / t3)
            exec2 += speedup_gt
            speedup_gt_rand = ((t1 - t4) / t1)
            speedup_base_rand = ((t3 - t4) / t3)
            #print("speedup compared to ground truth = ", speedup_gt)
            exec3 += speedup_base
            #!print("speedup compared to baseline O3 = ", speedup_base)
            #if ((abs(speedup_gt) > 2) or (abs(speedup_base) > 2)):
            #    bad.append(fn)
            exec4 += speedup_gt_rand
            exec5 += speedup_base_rand



        acc = acc / len(files) * 100.0
        exec1 = exec1 / len(files)
        exec2 = exec2 / len(files) * 100
        exec3 = exec3 / len(files) * 100
        exec4 = exec4 / len(files) * 100
        exec5 = exec5 / len(files) * 100

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
    print("reward_mean_list = ", reward_mean_list)
    print("reward_min_list = ", reward_min_list)
    print("reward_max_list = ", reward_max_list)
    print("total_loss_list = ", total_loss_list)
    print("accuracy_list = ", accuracy_list)
    print("exec2_list = ", exec2_list)
    print("exec3_list = ", exec3_list)
    print("exec4_list = ", exec4_list)
    print("exec5_list = ", exec5_list)


    #print("results = ", results)

    ray.shutdown()
