import torch
from torch.optim import Optimizer
import numpy as np 
import math, time

'''
Contains multiple optimizers for kin_xtrainer
'''

class KinBin(Optimizer):
    """
    Implements binary search KinBin (dub when land > launch, half when launch < land)
    """

    def __setstate__(self, state):
        super(KinBin, self).__setstate__(state)

    def __init__(self, params):
        default_dict = {'g': 1.0}
        super(KinBin, self).__init__(params, defaults = default_dict)

        # settings 
        self.g = default_dict['g']
        self.grad_dict = {} 
        self.params = params
        self.model_save_name = './'+str(round(time.time()))
        self.past_g = []
        
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

        # save model 
        # torch.save(model.state_dict(), self.model_save_name)

        # get norm of total old gradient 
        old_grad_tens = torch.Tensor().to('cuda')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                else:
                    old_grad_tens = torch.cat((old_grad_tens, p.grad.view(-1).to('cuda')))
        old_grad_norm = torch.norm(old_grad_tens, p = 2)
        
        # get time of impact 
        loss = closure()
        self.h = float(loss.item())
        self.vf = self.get_final_velocity(start_height = self.h)
        vf = float(self.vf)
        t_impact = self.time_of_impact(self.vf)

        old_h = float(closure().item())

        # get initial velocity (old_grad)
        new_location = {}
        old_grad_dict = {}
        group_index = 0
        for group in self.param_groups:
            param_index = 0
            old_grad_dict[group_index] = {}
            new_location[group_index] = {}
            for p in group['params']:
                if p.grad is None:
                    continue
                # store old grad 
                old_grad = p.grad/old_grad_norm 
                # update position 
                p.data.add_(-t_impact, old_grad) 
                param_index += 1
            group_index += 1

        
        # new loss
        loss = closure()
        new_h = float(closure().item())

        # double gravity 
        if new_h > old_h:
            self.g *= 2.0
        else:
            self.g *= 0.5

        return loss

class KinDub(Optimizer):
    """
    Classic Kinematics, but doubles g (w/o relaunch) whenever land h > launch h
    *** Saves past gs for later analysis
    """

    def __setstate__(self, state):
        super(KinDub, self).__setstate__(state)

    def __init__(self, params):
        default_dict = {'g': 1.0}
        super(KinDub, self).__init__(params, defaults = default_dict)

        # settings 
        self.g = default_dict['g']
        self.grad_dict = {} 
        self.params = params
        self.model_save_name = './'+str(round(time.time()))
        self.past_g = []
        
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

         # record past g 
        self.past_g.append(float(self.g))

        # get norm of total old gradient 
        old_grad_tens = torch.Tensor().to('cuda')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                else:
                    old_grad_tens = torch.cat((old_grad_tens, p.grad.view(-1).to('cuda')))
        old_grad_norm = torch.norm(old_grad_tens, p = 2)
        
        # get time of impact 
        loss = closure()
        self.h = float(loss.item())
        self.vf = self.get_final_velocity(start_height = self.h)
        vf = float(self.vf)
        t_impact = self.time_of_impact(self.vf)

        old_h = float(closure().item())

        # get initial velocity (old_grad)
        new_location = {}
        old_grad_dict = {}
        group_index = 0
        for group in self.param_groups:
            param_index = 0
            old_grad_dict[group_index] = {}
            new_location[group_index] = {}
            for p in group['params']:
                if p.grad is None:
                    continue
                # store old grad 
                old_grad = p.grad/old_grad_norm 
                # update position 
                p.data.add_(-t_impact, old_grad) 
                param_index += 1
            group_index += 1

        
        # new loss
        loss = closure()
        new_h = float(closure().item())

        # double gravity 
        if new_h > old_h:
            self.g *= 2

        return loss

