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
MEMORY_CAPCITY = 50000
EPISLON = 1
GAMMA = 0.99
LR_ALPHA = 0.001 
LR_BETA = 0.001 
N_STATES = 58
N_ACTIONS = 3
N_PARAM = 5
TARGET_PREDICT_ITER = 150
TOTAL_EPISODE = 20000
EPISODES = []
EP_REWARDS = []
lr_n = 0


####################  Parameterized DQN #################
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

class PDQN:
    def __init__(self):
        self.xnet = xNet(N_STATES, N_PARAM).to(device)
        self.eval_qnet, self.target_qnet = QNet(N_STATES+N_PARAM,N_ACTIONS).to(device), QNet(N_STATES+N_PARAM,N_ACTIONS).to(device)
        self.learn_step_counter = 0
        self.lr_n = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPCITY, N_STATES*2+2+N_PARAM))
        self.optimizer_q = torch.optim.Adam(self.eval_qnet.parameters(),lr=LR_ALPHA)
        self.optimizer_x = torch.optim.Adam(self.xnet.parameters(),lr=LR_BETA)
        self.criterion_q = nn.MSELoss()
        self.criterion_x = SumLoss()
    

    def choose_action(self, s, episode):
        s = torch.unsqueeze(torch.FloatTensor(s), 0).to(device)
        params = self.xnet(s)
        input_tensor = torch.cat([s,params],1)
        actions_value = self.eval_qnet(input_tensor)
        params = params.data.cpu().numpy()

        if np.random.uniform() > max(1 - episode*0.0002,0.1):
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        else:
            action = np.random.randint(0,N_ACTIONS)
            action = np.array([action])
            
        return action, params

    def store_transition(self, s, a, x, r, s_):
        x = torch.squeeze(torch.FloatTensor(x),0)
        transition = np.hstack((s, a, x, [r], s_))
        index = self.memory_counter % MEMORY_CAPCITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    
    def adjust_learning_rate(self, optimizer_q, optimizer_x):
        self.lr_n += 1
        lr = 0.001 * (math.exp(- self.lr_n / 15000))
        for param_group in optimizer_q.param_groups:
            param_group['lr'] = lr
        lr = 0.001 * (math.exp(- self.lr_n / 10000))
        for param_group in optimizer_x.param_groups:
            param_group['lr'] = lr
            
        '''
        self.lr_n += 1
        for param_group in optimizer_x.param_groups:
            param_group['lr'] = 0.001/self.lr_n
        if self.lr_n == 1:
            for param_group in optimizer_q.param_groups:
                param_group['lr'] = 0.001/self.lr_n
        else:
            for param_group in optimizer_q.param_groups:
                param_group['lr'] = 0.001/(self.lr_n*math.log(self.lr_n))
        '''

    def learn(self,episode):
        if self.learn_step_counter % TARGET_PREDICT_ITER == 0:
            self.target_qnet.load_state_dict(self.eval_qnet.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPCITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, : N_STATES]).to(device)
        b_a = torch.LongTensor(b_memory[:, N_STATES : N_STATES+1]).to(device)
        b_x = torch.FloatTensor(b_memory[:, N_STATES+1 : N_STATES+1+N_PARAM]).to(device)
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1+N_PARAM : N_STATES+2+N_PARAM]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)
        
        b_x_ = self.xnet(b_s_)
        
        q_eval = self.eval_qnet(torch.cat([b_s,b_x], 1)).gather(1, b_a)
        q_next = self.target_qnet(torch.cat([b_s_,b_x_],1)).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss_q = self.criterion_q(q_eval, q_target)
        
        with open('pdqn/loss_q.txt','a+') as fq:
            fq.write(str(loss_q.item())+'\n')
            fq.close()  

        self.adjust_learning_rate(optimizer_q = self.optimizer_q, optimizer_x = self.optimizer_x)

        self.optimizer_q.zero_grad()
        loss_q.backward()
        '''
        if episode % 50 == 0:
            print('**************************0')
            for name, parms in self.eval_qnet.named_parameters():
                print('-->name:', name, '-->grad_requires:', parms.requires_grad, '-->grad_value:', parms.grad)
        '''
        self.optimizer_q.step()

        x_all = self.xnet(b_s)
        input_xnet = torch.cat([b_s,x_all], 1)
        q_all = self.eval_qnet(input_xnet)
        loss_x = self.criterion_x(q_all,x_all)

        self.optimizer_x.zero_grad()
        
        with open('pdqn/loss_x.txt','a+') as fx:
            fx.write(str(loss_x.item())+'\n')
            fx.close()        
        
        loss_x.backward()
        '''
        if episode % 50 == 0:
            print('^^^^^^^^^^^^^^^^^^^^^^1')
            for name, parms in self.eval_qnet.named_parameters():
                print('-->name:', name, '-->grad_requires:', parms.requires_grad, '-->grad_value:', parms.grad)
            print('#######################2')
            for name, parms in self.xnet.named_parameters():
                print('-->name:', name, '-->grad_requires:', parms.requires_grad, '-->grad_value:', parms.grad)
        '''
        self.optimizer_x.step()

        
 
    def save_net(self, folder):
        torch.save(self.target_qnet, 'pdqn/'+folder+'/pdqn_target_qnet.pkl')
        torch.save(self.eval_qnet, 'pdqn/'+folder+'/pdqn_eval_qnet.pkl')
        torch.save(self.xnet, 'pdqn/'+folder+'/pdqn_xnet.pkl')

class SumLoss(nn.Module):
    def __init__(self):
        super(SumLoss, self).__init__()

    def forward(self, q_all , x_all):
        sigma_q = torch.sum(q_all,1)
        loss_x = -torch.mean(sigma_q) 
        '''
        penalty = 0
        for params in x_all:
            x = params.squeeze(0)
            if x[0] < -1:
                penalty += (-1-x[0])*(-1-x[0])
            if x[0] > 1:
                penalty += (1-x[0])*(1-x[0])
            if x[1] > 1:
                penalty += (x[1]-1)*(x[1]-1)
            if x[1] < -1:
                penalty += (-1-x[1])*(-1-x[1])
            if x[2] > 1:
                penalty += (x[2]-1)*(x[2]-1)
            if x[2] < -1:
                penalty += (-1-x[2])*(-1-x[2])
            if x[3] < 0:
                penalty += x[3]*x[3]
            if x[3] > 1:
                penalty += (x[3]-1)*(x[3]-1)
            if x[4] > 1:
                penalty += (x[4]-1)*(x[4]-1)
            if x[4] < -1:
                penalty += (-1-x[4])*(-1-x[4])
        penalty /= BATCH_SIZE
        '''
        return loss_x 


'''
reward 1
'''

def get_ball_dist_goal(sta):
    ball_proximity = sta[53]
    goal_proximity = sta[15]
    ball_dist = 1.0 - ball_proximity
    goal_dist = 1.0 - goal_proximity
    ball_ang_sin_rad = sta[51]
    ball_ang_cos_rad = sta[52]
    ball_ang_rad = math.acos(ball_ang_cos_rad)
    if ball_ang_sin_rad < 0:
        ball_ang_rad *= -1.
    goal_ang_sin_rad = sta[13]
    goal_ang_cos_rad = sta[14]
    goal_ang_rad = math.acos(goal_ang_cos_rad)
    if goal_ang_sin_rad < 0:
        goal_ang_rad *= -1.
    alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad, goal_ang_rad)
    ball_dist_goal = math.sqrt(
        ball_dist * ball_dist + goal_dist * goal_dist - 2. * ball_dist * goal_dist * math.cos(alpha))
    return ball_dist_goal


# there's explicit codes about reward in src/hfo_game.cpp
def getReward(old_state, current_state, get_kickable_reward, status,hfo_Env):
    r = 0
    kickable = current_state[12]
    old_kickable = old_state[12]

    ball_prox_delta = current_state[53] - old_state[53]  # ball_proximity - old_ball_prox
    kickable_delta = kickable - old_kickable
    ball_dist_goal_delta = get_ball_dist_goal(current_state) - get_ball_dist_goal(old_state)
    player_on_ball = hfo_env.playerOnBall()
    our_unum = hfo_env.getUnum()

    # move to ball reward
    if player_on_ball.unum < 0 or player_on_ball.unum == our_unum:
        r += ball_prox_delta
    # if kickable_delta >= 1 and (not get_kickable_reward):
    if kickable_delta >= 1:
        r += 1.0
        get_kickable_reward = True

    # kick to goal reward
    if player_on_ball.unum == our_unum:
        r -= 3 * ball_dist_goal_delta
    # elif get_kickable_reward:  # we have passed to teammate
    #     r -= 3 * 0.2 * ball_dist_goal_delta

    # EOT reward
    if status == GOAL:
        if player_on_ball.unum == our_unum:
            r += 5
        else:
            r += 1
    elif status == CAPTURED_BY_DEFENSE:
        r += 0

    return r, get_kickable_reward

'''
reward 2
'''
def kick_to_goal_reward(s):
    ball_proximity = s[53]
    goal_proximity = s[15]
    ball_dist = 1.0 - ball_proximity
    goal_dist = 1.0 - goal_proximity
    ball_ang_sin_rad = s[51]
    ball_ang_cos_rad = s[52]
    ball_ang_rad = math.acos(ball_ang_cos_rad)
    if ball_ang_sin_rad < 0:
        ball_ang_rad *= -1.0
    goal_ang_sin_rad = s[13]
    goal_ang_cos_rad = s[14]
    goal_ang_rad = math.acos(goal_ang_cos_rad)
    if goal_ang_sin_rad < 0:
        goal_ang_rad *= -1.0
    alpha = max(ball_ang_rad, goal_ang_rad) \
        - min(ball_ang_rad, goal_ang_rad)
    ball_dist_goal = math.sqrt(ball_dist*ball_dist + goal_dist*goal_dist - \
                        2.0*ball_dist*goal_dist*math.cos(alpha))
    return ball_dist_goal

def EOT_reward(status):
    if status == GOAL:
        return 5
    else:
        return 0
#_______________________train_____________________________


pdqn = PDQN()
for episode in range(TOTAL_EPISODE):
    hfo_env = HFOEnvironment()
    hfo_env.connectToServer(LOW_LEVEL_FEATURE_SET,
                            'bin/teams/base/config/formations-dt', 6000,
                            'localhost', 'base_left', False)
    for episode in range(TOTAL_EPISODE):
        status = IN_GAME
        first_kick = 1
        ep_r = 0
        iteration = 0
        got_kickable_r = False

        while status == IN_GAME:
            iteration += 1
            s = hfo_env.getState()
            s = s[:58]
            a,x = pdqn.choose_action(s,episode)
            if a == 0:
                hfo_env.act(a,x[0][0]*100,x[0][1]*180)
            elif a == 1:
                hfo_env.act(a,x[0][2]*180)
            elif a == 2:
                hfo_env.act(KICK,x[0][3]*100,x[0][4]*180)
            status = hfo_env.step()
            s_ = hfo_env.getState()
            s_ = s_[:58]

            # reward 1
            #r, got_kickable_r = getReward(s, s_, got_kickable_r, status,hfo_env)
            #r = s[9] + s[12] + 2*(s_[53] - s[53])
            kickable = s_[12]
            if kickable == 1:
                first_kick = 0
                r = s[53] - s_[53] + first_kick + 3.0*(kick_to_goal_reward(s) - \
                        kick_to_goal_reward(s_)) + EOT_reward(status)
            else:
                r = s[53] - s_[53] + 3.0*(kick_to_goal_reward(s) - \
                        kick_to_goal_reward(s_)) + EOT_reward(status)
                        
            pdqn.store_transition(s, a, x, r, s_)
            
            ep_r += r
                
            if pdqn.memory_counter > MEMORY_CAPCITY:
                pdqn.learn(episode)
            
        ep_r = ep_r / iteration
        with open('pdqn/ep_rewards.txt','a+') as f:
            f.write(str(ep_r)+'\n')
            f.close()
        
        print(('Episode %d ended with %s'%(episode, hfo_env.statusToString(status))))
        
        if(episode == 100):
            pdqn.save_net('100')
        
        if(episode == 10000):
            pdqn.save_net('10000')

        if(episode == 19999):
            pdqn.save_net('19999')
            
        if status == SERVER_DOWN:
            pdqn.save_net('serverdown')
            hfo_env.act(QUIT)
            exit()