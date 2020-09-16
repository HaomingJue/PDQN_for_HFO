#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hfo import *
import math
import matplotlib.pyplot as plt

####################  hyper parameters ##################
BATCH_SIZE = 64
MEMORY_CAPCITY = 500000
EPISLON = 1
GAMMA = 0.99
LR_ALPHA = 0.001 
LR_BETA = 0.001 
N_STATES = 58
N_ACTIONS = 3
N_PARAM = 5
TARGET_PREDICT_ITER = 150
TOTAL_EPISODE = 3000000
EPISODES = []
EP_REWARDS = []
lr_n = 0


####################  parameterized DQN #################
device = 'cuda'if torch.cuda.is_available() else 'cpu'



class QNet(nn.Module):
    def __init__(self, in_dim, a_dim):
        super(QNet,self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(256,128)
        self.fc2.weight.data.normal_(0,0.1)
        self.fc3 = nn.Linear(128,64)
        self.fc3.weight.data.normal_(0,0.1)
        #self.fc4 = nn.Linear(64,64)
        #self.fc4.weight.data.normal_(0,0.1)
        self.out = nn.Linear(64,a_dim)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        q = F.relu(self.out(x))
        return q


class xNet(nn.Module):
    def __init__(self, s_dim, param_dim):
        super(xNet,self).__init__()
        self.fc1 = nn.Linear(s_dim,256)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(256,128)
        self.fc2.weight.data.normal_(0,0.1)
        self.fc3 = nn.Linear(128,64)
        self.fc3.weight.data.normal_(0,0.1)
        self.out = nn.Linear(64,param_dim)
        self.out.weight.data.normal_(0,0.1)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        param = F.tanh(self.out(x))
        return param



xnet = torch.load('pdqn/pdqn_xnet.pkl')
eval_qnet = torch.load('pdqn/pdqn_eval_qnet.pkl')

def choose_action( s):
    s = torch.unsqueeze(torch.FloatTensor(s), 0).to(device)
    params = xnet(s)
    input_tensor = torch.cat([s,params],1)
    actions_value = eval_qnet(input_tensor)
    params = params.data.cpu().numpy()
    action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            
    return action, params

#_______________________train_____________________________

    

for episode in range(TOTAL_EPISODE):
    hfo_env = HFOEnvironment()
    hfo_env.connectToServer(LOW_LEVEL_FEATURE_SET,
                            'bin/teams/base/config/formations-dt', 6000,
                            'localhost', 'base_left', False)
    for episode in range(TOTAL_EPISODE):
        status = IN_GAME
        first_kick = True
        ep_r = 0
        iteration = 0
        got_kickable_r = False

        while status == IN_GAME:
            iteration += 1
            s = hfo_env.getState()
            s = s[:58]
            a,x = choose_action(s)
            if a == 0:
                hfo_env.act(a,x[0][0]*100,x[0][1]*180)
            elif a == 1:
                hfo_env.act(a,x[0][2]*180)
            elif a == 2:
                hfo_env.act(KICK,x[0][3]*100,x[0][4]*180)
            status = hfo_env.step()
            s_ = hfo_env.getState()


        

        print(('Episode %d ended with %s'%(episode, hfo_env.statusToString(status))))
    
        if status == SERVER_DOWN:
            hfo_env.act(QUIT)
            exit()