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
from envs.neurovec_gnn import NeuroVectorizerEnv
from ray.tune.registry import register_env
from ray.tune.logger import TBXLogger
from gnn import GCNClassifier
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("gnn_model", GCNClassifier)
import argparse
import json
import torch

from ray.rllib.agents.registry import get_trainer_class


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
    "--stop-timesteps",
    type=int,
    default=10000,
    help="Number of timesteps to train before we do inference.",
)

parser.add_argument(
    "--explore-during-inference",
    action="store_true",
    help="Whether the trained policy should use exploration during action "
    "inference.",
)

register_env("autovec_gnn", lambda config:NeuroVectorizerEnv(config))

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    env_config = {'dirpath':'./training_data','new_rundir':'./new_garbage','graphpath':'./training_edge_list'}

    config = {
        #"sample_batch_size": 1,
        "train_batch_size": 500,
        "sgd_minibatch_size": 20,
        "num_sgd_iter":20,
        #"lr":5e-5,
        #"vf_loss_coeff":0.5,
        "env": "autovec_gnn",
        "horizon":  1,
        "num_gpus": 0,
        "model": {
            "custom_model": "gnn_model",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {"hidden_dim": 128, "num_layers": 2},
        },
        "num_workers": 1,
        "env_config":env_config,
        "framework": args.framework,
        # Run with tracing enabled for tfe/tf2?
        "eager_tracing": args.eager_tracing,
    }

    tune.run("PPO",
            #restore = "~/ray_results/PPO_*/checkpoint_240/checkpoint-240",
            checkpoint_freq  = 1,
            name = "neurovectorizer_train",
            stop = {
                "timesteps_total": args.stop_timesteps,
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

    #while num_episodes < args.num_episodes_during_inference:
    # iterate the dataset
    with open("features.json") as f:
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
        obs = []        
        #print("feat = ", feat)
        print("len = ", len(feat))
        obs.append(torch.IntTensor(feat[0:200]).numpy())
        obs.append(torch.IntTensor(feat[200:400]).numpy())
        obs.append(torch.IntTensor(feat[400:600]).numpy())
        obs.append(torch.FloatTensor(feat[600:800]).numpy())
        #print("obs = ", obs)
        # Compute an action (`a`).
        a = trainer.compute_single_action(
            observation=obs,
            explore=args.explore_during_inference,
            policy_id="default_policy",  # <- default value
        )
        print("action = ", a)
        if (label[0] == a[0] and label[1] == a[1]):
            acc += 1
        # Send the computed action `a` to the env.
        _, reward, _, _ = env.step(a)
        print("reward = ", reward)
        rewards.append(reward)

    acc = acc / len(files) * 100.0
    print("accuracy = ", acc)

    with open('rewards_neuro.json', 'w') as f:
        json.dump(rewards, f) 

    ray.shutdown()
