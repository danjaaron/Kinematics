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
        self.past_grad_norm = []
        
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
        self.past_grad_norm.append(float(old_grad_norm))
        
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
        self.past_grad_norm = []
        
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
        self.past_grad_norm.append(float(old_grad_norm))
        
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

class KinProg(Optimizer):
    """
    Kin Progressive from Brijen analysis test.py
    """

    def __setstate__(self, state):
        super(KinProg, self).__setstate__(state)

    def __init__(self, params):
        default_dict = {'g': 1e-10}
        super(KinProg, self).__init__(params, defaults = default_dict)

        # settings 
        self.g = default_dict['g']
        self.grad_dict = {} 
        self.params = params
        self.model_save_name = './'+str(round(time.time()))
        self.past_g = []
        self.past_grad_norm = []
        self.progress = [0.]
        self.scale_factor = 10.0
        
    def step(self, closure, model):
        """
        Performs a single optimization step.

        Arguments:
            loss (req): PyTorch loss Tensor at each step 
        """

        print(self.g)

        
        self.past_g.append(float(self.g))

        # get norm of total old gradient 
        old_grad_tens = torch.Tensor().to('cuda')
        cloned_data = {}
        group_index = 0
        for group in self.param_groups:
            param_index = 0
            cloned_data[group_index] = {}
            for p in group['params']:
                if p.grad is None:
                    continue
                else:
                    old_grad_tens = torch.cat((old_grad_tens, p.grad.view(-1).to('cuda')))
                    # clone data for curr prog
                    cloned_data[group_index][param_index] = p.data.clone().detach()
                param_index += 1
            group_index += 1
        old_grad_norm = torch.norm(old_grad_tens, p = 2)
        self.past_grad_norm.append(float(old_grad_norm))
        

        l0 = float(closure().item())
        t_impact = np.sqrt(2.0 * float(l0)/self.g)

        # update x
        group_index = 0
        for group in self.param_groups:
            param_index = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                v = -p.grad/old_grad_norm 
                p.data = cloned_data[group_index][param_index] + t_impact*v
                param_index += 1
            group_index += 1

        lf = float(closure().item())

        prev_p = self.progress[-1] 
        curr_p = float(l0 - lf)

        while ((curr_p < 0.0)):
            print("repeat")
            self.g *= self.scale_factor
            t_impact = np.sqrt(2.0 * float(l0)/self.g)
            # update x
            group_index = 0
            for group in self.param_groups:
                param_index = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    v = -p.grad/old_grad_norm 
                    p.data = cloned_data[group_index][param_index] + t_impact*v
                    param_index += 1
                group_index += 1
            # get new progress
            lf = float(closure().item())
            curr_p = float(l0 - lf)

        self.progress.append(curr_p)

        if lf > l0 - lf/float(self.g):
            print('g +')
            self.g *= self.scale_factor

        return lf
'''
def p_optimize(x0, f, df, df_normalized, T=iters, verbose = False):
    # increase g if less progress is made than last step
    g = 1e-10
    scale_factor = 10.0
    larr, ssarr, garr = [], [], []
    progress = [0.]
    for i in range(T):
        
        l0 = f(x0)
        step_size = np.sqrt(2.0 * float(l0)/g)

        
        
        larr.append(l0)
        garr.append(g)
        ssarr.append(step_size)

        print(df_normalized(x0))
        print(df(x0))

        xf = x0 - step_size * df_normalized(x0)

        lf = f(xf) 

        prev_p = progress[i] 
        curr_p = float(l0 - lf) # progress in this step

        while ((curr_p < 0.0)): # and (g <= abs(lf/l0))): 
            g *= scale_factor
            step_size = np.sqrt(2.0 * float(l0)/g)
            xf = x0 - step_size * df_normalized(x0)
            lf = f(xf) 
            curr_p = float(l0 - lf)
            if verbose:
                print("repeated ", lf, curr_p, prev_p, g)
        curr_p = float(l0 - lf) # progress in this step
        progress.append(curr_p)

        # # increase g if last step made more progress
        # if (prev_p < curr_p):
        #   g *= 10.
        # else:
        #   g *= 0.1
        
        if lf >= l0 - lf/float(g):
            if verbose:
                print('g +')
            g *= scale_factor

        if verbose:
            print(i, lf, x0, xf, curr_p, prev_p, g)
        x0 = xf

    return larr, garr
'''