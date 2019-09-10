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
        default_dict = {'g': 1.0}
        super(Kinematics, self).__init__(params, defaults = default_dict)

        # settings 
        self.g = default_dict['g']
        self.grad_dict = {} 
        self.params = params
        self.model_save_name = './'+str(round(time.time()))
        self.past_g = []
        self.v0 = 0. # scalar downward velocity
        

    def time_of_impact(self, uf_in):
        """
        Computes time of impact from current height (loss) to loss = 0
        under gravitational acceleration. 
        """
        a = self.g
        t = (float(uf_in) - self.v0)/float(a)
        return t

    def get_final_velocity(self, start_height):
        """
        Computes downward velocity at time of impact (loss = 0).
        """
        uf = math.sqrt(self.v0**2 + 2.0*abs(self.g)*abs(start_height))
        return uf
        
    def step(self, closure, model):
        """
        Performs a single optimization step.

        Arguments:
            loss (req): PyTorch loss Tensor at each step 
        """

        v0 = float(self.v0)

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

        # drift weights with t_impact
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
                old_grad_dict[group_index][param_index] = old_grad
                # update position 
                p.data.add_(-t_impact, old_grad) 
                new_location[group_index][param_index] = p.data.clone().detach() # location after launch
                param_index += 1
            group_index += 1

         # new loss
        loss = closure()
        new_h = float(closure().item())
        
        
        # adjust vf based on actual landing height
        if new_h < old_h:
            new_vf = np.sqrt(v0**2 + 2.*self.g*(old_h - new_h)) #v0 + self.g*t_impact
            print("vf: {} -> {}".format(vf, new_vf))
            self.vf = new_vf

        # set next v0
        self.v0 = self.vf

        print("new v0 {}".format(self.v0))

        return loss
