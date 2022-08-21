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
import gym
from gym import spaces
import pickle
import numpy as np
import re
import os
import logging

from extractor_c import CExtractor
from config import Config
from my_model import Code2VecModel
from path_context_reader import EstimatorAction

from utility_neuro import get_bruteforce_runtimes, get_snapshot_from_code, get_runtime, get_vectorized_codes, get_encodings_from_local, MAX_LEAF_NODES, pragma_line
from lore_utility import init_runtimes_dict, get_O3_runtimes, load_observations_dict

logger = logging.getLogger(__name__)

#NeuroVectorizer RL Environment
class NeuroVectorizerEnv(gym.Env):
    def __init__(self, env_config):
        self.dim = 126
        self.type = 'max'
        self.init_from_env_config(env_config)
        self.copy_train_data()
        self.parse_train_data()
        #self.config_AST_parser()
        self.init_RL_env()
        # Keeps track of the file being processed currently.
        self.current_file_idx = 0
        # Keeps track of the current loop being processed currently in that file.
        self.current_pragma_idx = 0
        '''Runtimes dict to stored programs the RL agent explored.
         This saves execution and compilation time due to dynamic programming.'''
        #print("&&&&&&& before init_runtimes_dict")
        self.runtimes = init_runtimes_dict(self.new_testfiles,self.num_loops,
                        len(self.vec_action_meaning),len(self.interleave_action_meaning))
        #print("self.runtimes = ", self.runtimes)
        #quit()
        '''Observations dictionary to store AST encodings of programs explored by the RL agent. 
        It saves time when the RL agent explores a program it explored before.
        It is also initialized from obs_encodings.pkl file to further save time.''' 
        #self.obs_encodings = get_encodings_from_local(self.new_rundir)
        #print("&&&&&&&&& before load_obser_dict")
        self.obs_encodings = load_observations_dict(self.new_rundir, 'loregcc_graphsage_feat2_128.json') 
        #self.obs_encodings = load_observations_dict(self.new_rundir, 'features2_graphsage_gcn128.json')
        #print("obs = ", self.obs_encodings)
        #quit()
        if self.compile:
            # stores the runtimes of O3 to compute the RL reward and compared to -O3.
            #print("&&&&&& before get O3 runtimes")
            self.O3_runtimes=get_O3_runtimes(self.new_rundir, self.new_testfiles)
            #print("@@@@@@@@@@@@@after get O3 runtimes")
    
    def init_from_env_config(self,env_config):
        '''Receives env_config and initalizes all config parameters.'''
        # dirpath is the path to the train data.
        self.dirpath = env_config.get('dirpath')
        # new_rundir is the directory to create and copy the train data to.
        self.new_rundir = env_config.get('new_rundir') + '_' + str(os.getpid())
        # whether or not in inference mode
        self.inference_mode = env_config.get('inference_mode', False)
        if self.inference_mode:
            # Used in inference mode to print current geomean improvement.
            self.improvements=[]
        '''Whether to compile the progams or not, generally turned off 
        in inference mode when it is not clear how to compile (e.g., requires make)
        '''
        self.compile = env_config.get('compile', True) 
        #if your code is not structured like the given training data.
        self.new_train_data = env_config.get('new_train_data',False) 
    
    def copy_train_data(self):
        '''Copy the train data to a new directory.
        used to inject pragmas in the new files,
        without modifying original files.
        '''
        if not os.path.exists(self.new_rundir):
            print('creating '+self.new_rundir+' directory')
            os.mkdir(self.new_rundir)

        cmd = 'cp -r ' +self.dirpath+'/* ' +self.new_rundir
        print('running:',cmd)
        os.system(cmd)
    
    def init_RL_env(self):
        #print("***********init_RL_env")
        ''' Defines the reinforcement leaning environment.
        Modify to match your hardware and programs.
        '''
        self.vec_action_meaning = [1,2,4,8,16] # TODO: change this to match your hardware
        self.interleave_action_meaning=[1,2,4,8,16] # TODO: change this to match your hardware
        self.action_space = spaces.Tuple([spaces.Discrete(len(self.vec_action_meaning)),
                                        spaces.Discrete(len(self.interleave_action_meaning))])
        '''The observation space is bounded by the word dictionary 
        the preprocessing generated.
        self.observation_space = spaces.Tuple(
                                 [spaces.Box(0,3000,shape=(self.config.MAX_CONTEXTS,),dtype = np.int32,)]
                                 +[spaces.Box(0,3000,shape=(self.config.MAX_CONTEXTS,),dtype = np.int32,)]
                                 +[spaces.Box(0,3000,shape=(self.config.MAX_CONTEXTS,),dtype = np.int32,)]
                                 +[spaces.Box(-1,2,shape=(self.config.MAX_CONTEXTS,),dtype = np.float32)]
                                 )
        '''
        #self.observation_space = spaces.Box(-3000,3000,shape=((self.dim+2),),dtype = np.float32,)
        self.observation_space = spaces.Box(-1,1.1,shape=((self.dim+2),),dtype = np.float32,)
    def parse_train_data(self):
        #print("******** In parse train data")
        ''' Parse the training data. '''
        self.orig_train_files = [os.path.join(root, name)
             for root, dirs, files in os.walk(self.new_rundir)
             for name in files
             if name.endswith(".c") and not name.startswith('header.c') 
             and not name.startswith('aux_AST_embedding_code.c')]
        # copy testfiles
        self.new_testfiles = list(self.orig_train_files)
        tmp = []
        self.num_loops = {}
        for testfile in self.new_testfiles:
            name = testfile.split('/', 2)[-1][:-2].rpartition('_')[0].rpartition('/')[0]
            #print("testfile = ", testfile)
            #print("name = ", name)
            
            tmp.append(name)
            self.num_loops[name] = 1

        self.new_testfiles = tmp




        #print("nre files = ", self.new_testfiles)
        # parse the code to detect loops and inject commented pragmas.  
        #self.loops_idxs_in_orig,self.pragmas_idxs,self.const_new_codes,self.num_loops,self.const_orig_codes \
        #= get_vectorized_codes(self.orig_train_files,self.new_testfiles)
        # to operate only on files that have for loops.
        #self.new_testfiles = list(self.pragmas_idxs.keys())
 
    def config_AST_parser(self):
        '''Config the AST tree parser.'''
        self.config = Config(set_defaults=True, load_from_args=False, verify=True)
        self.code2vec = Code2VecModel(self.config)
        self.path_extractor = CExtractor(self.config,clang_path=os.environ['CLANG_PATH'],max_leaves=MAX_LEAF_NODES)
        self.train_input_reader = self.code2vec._create_data_reader(estimator_action=EstimatorAction.Train)
    
    def get_reward(self,current_filename,VF_idx,IF_idx):
        '''Calculates the RL agent's reward. The reward is the 
        execution time improvement after injecting the pragma
        normalized to -O3.'''
        #print("in get reward, current_filename = ", current_filename)
        if self.compile:
            if self.runtimes[current_filename][self.current_pragma_idx][VF_idx][IF_idx]:
                runtime = self.runtimes[current_filename][self.current_pragma_idx][VF_idx][IF_idx]
                #print("1")
            else:      
                #print("2")      
                #runtime = get_runtime(self.new_rundir,new_code,current_filename)
                self.runtimes[current_filename][self.current_pragma_idx][VF_idx][IF_idx]=runtime
            #print("runtime = ", runtime)
            if self.O3_runtimes[current_filename]==None:
                reward = 0
                logger.warning('Program '+current_filename+' does not compile in two seconds.'+
                               ' Consider removing it or increasing the timeout parameter'+
                               ' in utility.py.')
            elif runtime==None:
                #penalizing for long compilation time for bad VF/IF
                reward = -9
            else:    
                #print("O3 = ", self.O3_runtimes[current_filename], ", pred = ", runtime)
                reward = (self.O3_runtimes[current_filename]-runtime)/self.O3_runtimes[current_filename]
            # In inference mode and finished inserting pragmas to this file.
            if self.inference_mode and self.current_pragma_idx+1 == self.num_loops[current_filename]:
                improvement = self.O3_runtimes[current_filename]/runtime
                self.improvements.append(improvement)
                geomean = 1
                for imp in self.improvements:
                    geomean = geomean * (imp**(1/len(self.improvements))) 
                print('benchmark: ',current_filename,'O3 runtime: ', 
                      self.O3_runtimes[current_filename], 'RL runtime: ', runtime,
                      'improvement:',str(round(improvement,2))+'X',
                      'improvement geomean so far:',str(round(geomean,2))+'X')
            VF = self.vec_action_meaning[VF_idx]
            IF = self.interleave_action_meaning[IF_idx]
            opt_runtime_sofar=self.get_opt_runtime(current_filename,self.current_pragma_idx)
            logger.info(current_filename+' runtime '+str(runtime)+' O3 ' + 
                        str(self.O3_runtimes[current_filename]) +' reward '+str(reward)+
                        ' opt '+str(opt_runtime_sofar)+" VF "+str(VF)+" IF "+str(IF))
        else:
            # can't calculate the reward without compile/runtime.
            reward = 0
        #print("exiting get reward")
        return reward

    def get_opt_runtime(self,current_filename,current_pragma_idx):
        min_runtime = float('inf')
        for VF_idx in self.runtimes[current_filename][self.current_pragma_idx]:
            for IF_idx in VF_idx:
                if IF_idx:
                    min_runtime = min(min_runtime,IF_idx)
        return min_runtime
                
    def reset(self):
        ''' RL reset environment function. '''
        #print("@@@@@@@@@@ in reset ")
        current_filename = self.new_testfiles[self.current_file_idx]
        #this make sure that all RL pragmas remain in the code when inferencing.
        #if self.current_pragma_idx == 0 or not self.inference_mode:
        #    self.new_code = list(self.const_new_codes[current_filename])
        return self.get_obs(current_filename,self.current_pragma_idx)

    def get_obs(self,current_filename,current_pragma_idx):
        '''Given a file returns the RL observation.
           Change this if you want other embeddings.'''
        #print("@@@@@@@@@@@ in get_obs")
        #print("@@@@@@@@@@ current filename = ", current_filename)
        #for f in self.obs_encodings:
        #    print("########## f = ", f)
        #Check if this encoding already exists (parsed before).
        try:
            #print("keys = ", self.obs_encodingd.keys())
            #print("current_filename = ", current_filename)
            return self.obs_encodings[current_filename][current_pragma_idx]
        except:
            pass
        #print("111111111111111")
        #for key in self.obs_encodings.keys():
        #    print("@@@@@@@@@key = ", key)
        #print("@@@@@@@@@@@@@@@@@current_filename = ", current_filename)

        assert False, "cannot reach here"
        # To get code for files not in the dataset.
        if self.new_train_data:
            code=get_snapshot_from_code(self.const_orig_codes[current_filename],
                                        self.loops_idxs_in_orig[current_filename][current_pragma_idx])
        else:
            code=get_snapshot_from_code(self.const_orig_codes[current_filename])

        input_full_path_filename=os.path.join(self.new_rundir,'aux_AST_embedding_code.c')
        loop_file=open(input_full_path_filename,'w')
        loop_file.write(''.join(code))
        loop_file.close()
        try:
            train_lines, hash_to_string_dict = self.path_extractor.extract_paths(input_full_path_filename)
        except:
            print('Could not parse file',current_filename, 'loop index',current_pragma_idx,'. Try removing it.')
            raise 
        dataset  = self.train_input_reader.process_and_iterate_input_from_data_lines(train_lines)
        obs = []
        tensors = list(dataset)[0][0]
        import tensorflow as tf
        for tensor in tensors:
            with tf.compat.v1.Session() as sess: 
                sess.run(tf.compat.v1.tables_initializer())
                obs.append(tf.squeeze(tensor).eval())

        if current_filename not in self.obs_encodings:
            self.obs_encodings[current_filename] = {}
        self.obs_encodings[current_filename][current_pragma_idx] = obs
        return obs

    def step(self,action):
        '''The RL environment step function. Takes action and applies it as
        VF/IF pragma for the parsed loop.'''
        #print("@@@@ in step")
        done = True # RL horizon = 1 
        action = list(np.reshape(np.array(action),(np.array(action).shape[0],)))
        VF_idx = action[0]
        IF_idx = action[1]
        VF = self.vec_action_meaning[VF_idx]
        IF = self.interleave_action_meaning[IF_idx]
        current_filename = self.new_testfiles[self.current_file_idx]
        #self.new_code[self.pragmas_idxs[current_filename][self.current_pragma_idx]] = pragma_line.format(VF,IF)
        reward = self.get_reward(current_filename,VF_idx,IF_idx)
        #print("VF",VF,"IF",IF)
        #print('filename = ', current_filename, ', reward = ', reward, "VF/IF = ", VF, IF)
        #print("222222222222")
        
        obs = self.get_obs(current_filename,0)
        self.current_file_idx = (1 + self.current_file_idx) % 8603

        return obs,reward,done,{}
