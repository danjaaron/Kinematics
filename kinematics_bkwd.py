import torch
from torch.optim import Optimizer
import numpy as np 
import math, time

class Kinematics(Optimizer):
    """
    Implements kinematics optimization method with forward collisions 
    and momentum conservation. 

    Uses EMA to adjust g adaptively 
    """

    def __setstate__(self, state):
        super(Kinematics, self).__setstate__(state)

    def __init__(self, params):
        default_dict = {'g': 9.81}
        super(Kinematics, self).__init__(params, defaults = default_dict)

        # settings 
        self.g = default_dict['g']
        self.v_dict = {} 
        self.params = params
        self.model_save_name = './'+str(round(time.time()))
        self.past_g = []
        self.last_h = 0
        self.last_t = 0
        

    def time_of_impact(self, uf_in):
        """
        Computes time of impact from current height (loss) to loss = 0
        under gravitational acceleration. 
        """
        a = self.g
        t = (float(uf_in))/float(a)
        # print("calc t impact: ", uf_in, self.v0)
        # self.v0 = uf_in
        return t

    def get_final_velocity(self, start_height):
        """
        Computes downward velocity at time of impact (loss = 0).
        """
        # print(self.v0)
        uf = math.sqrt(2.0*abs(self.g)*abs(start_height))
        # print(uf)
        return uf
        
    def step(self, closure, model):
        """
        Performs a single optimization step.

        Arguments:
            loss (req): PyTorch loss Tensor at each step 
        """

        # get time of impact 
        loss = closure()
        loss.backward()
        self.h = float(loss.item())
        # loss.backward(retain_graph = True)
        # loss.backward()

        # compute current velocity norm
        v_tens = torch.Tensor().to('cuda')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                else:
                    v_tens = torch.cat((v_tens, p.grad.view(-1).to('cuda')))
        v_norm = torch.norm(v_tens, p = 2)
        
        

        # if self.last_h > 0:
        #     h = (self.last_h + self.last_h)/2.
        # else:
        #     h = self.h
        
        # if self.last_h > 0:
        
        #     self.g += (self.h - self.last_h)

        # if self.last_t > 0 and self.last_h > 0:
        #     self.g  += (self.h - self.last_h)
        # self.g = max(self.g, 1e-5)
        h = abs(self.h - self.last_h)#.h #(self.h + self.last_h)/2.0

        self.vf = self.get_final_velocity(start_height = h)
        vf = float(self.vf)
        t_impact = self.time_of_impact(self.vf)

        # self.last_h += h

        # drift v over t_impact
        new_v_dict = {}
        group_index = 0
        for group in self.param_groups:
            param_index = 0
            new_v_dict[group_index] = {}
            for p in group['params']:
                if p.grad is None:
                    continue
                # current velocity
                v = p.grad/v_norm
                new_v_dict[group_index][param_index] = v.clone().detach()
                # avg with previous velocity 
                if self.v_dict:
                    p.data.add_(-t_impact, (self.v_dict[group_index][param_index] + v)/(2.0))
                else:
                    p.data.add_(-t_impact, v)
                param_index += 1
            group_index += 1

        # new_h = closure().item()

        # if new_h > self.h:
        #     self.g *= 2
        # self.g += 1

        self.last_h = float(self.h)
        self.last_t = float(t_impact)

        # update v dict 
        self.v_dict = new_v_dict.copy()
      
        return loss
