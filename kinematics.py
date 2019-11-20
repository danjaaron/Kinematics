import torch
from torch.optim import Optimizer
import numpy as np 
import math, time
import copy
from copy import deepcopy


class Kin(Optimizer):

    alias = 'K'

    def __init__(self, params):
        self.g = 1.0
        default_dict = {'g': float(self.g)}
        # print(super(self.__class__)
        super(Kin, self).__init__(params, defaults = default_dict)

    def __setstate__(self, state):
        super(Kin, self).__setstate__(state)

    def get_grad_norm(self):
        """ Get norm of gradient
        """
        flat_grad = torch.Tensor().to('cuda')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                else:
                    flat_grad = torch.cat((flat_grad, p.grad.view(-1).to('cuda')))
        grad_norm = torch.norm(flat_grad, p = 2)
        return grad_norm


    def get_step_size(self, loss):
        return np.sqrt(2.0*loss/self.g)

    def update_params(self, step_size, grad_norm):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(-step_size, p.grad / grad_norm)

    def update_g(self, h0, hf):
        if hf > h0:
            self.g *= 2.0           

    def step(self, closure, model):
        """ Update parameters 
        """ 

        self.closure = closure

        h0 = closure().item()

        grad_norm = self.get_grad_norm()
        step_size = self.get_step_size(h0)
        self.update_params(step_size, grad_norm)

        hf = closure().item()

        self.update_g(h0, hf)

        return h0



class KinMnm(Kin):

    alias = 'M'

    def __init__(self, params):
        super(KinMnm, self).__init__(params)
        self.v = {}
        self.t = 0
        self.b1 = 0.9
        for (gidx, group) in enumerate(self.param_groups):
                self.v[gidx] = dict()
                for (pidx, p) in enumerate(group['params']):
                    self.v[gidx][pidx] = dict()
                    self.v[gidx][pidx]['m1'] = 0.
                    self.v[gidx][pidx]['g'] = 1.0


    def update_params(self, step_size, grad_norm):
        
        self.t += 1
        for (gidx, group) in enumerate(self.param_groups):
            for (pidx, p) in enumerate(group['params']):
                if p.grad is None:
                    continue
                else:

                    group = self.v[gidx][pidx]

                    if self.t == 1:
                        group['m1'] = torch.zeros_like(p.data)
                        group['m2'] = torch.zeros_like(p.data)


                    h0 = self.closure().item()

                    t_flight = math.sqrt(2.0*h0/group['g'])

                    p.data.add_(-t_flight*group['m1'])

                    group['m1'] = (1 - self.b1)*group['m1'] + self.b1*(p.grad / grad_norm)

                    hf = self.closure().item()

                    if hf > h0:
                        # p_norm = torch.norm(p.grad.view(-1), p = 2).item()
                        group['g'] *= 2



    def update_g(self, h0, hf):
        if False:
            self.g *= 2.0



class KinAdg(Kin):
    """ Kinematics with adaptive (per-group) g value
    ... double g for each group if loss goes up after group is updated
    """

    alias = 'G'

    def __init__(self, params):
        super(KinAdg, self).__init__(params)
        self.g_dict = {}
        self.v = {}
        self.beta = 0.8
        for (gidx, group) in enumerate(self.param_groups):
            self.g_dict[gidx] = dict()
            self.v[gidx] = dict()
            for (pidx, p) in enumerate(group['params']):
                self.g_dict[gidx][pidx] = 1.0
                self.v[gidx][pidx] = 0.0


    def update_params(self, closure, grad_norm):
        for (gidx, group) in enumerate(self.param_groups):
            for (pidx, p) in enumerate(group['params']):
                if p.grad is None:
                    continue
                h0 = closure().item()
                step_size = np.sqrt(2.0 * h0 / self.g_dict[gidx][pidx]) # time 
                dp = -step_size * p.grad / grad_norm
                p.data.add_(dp)
                hf = closure().item()

                p_norm = torch.norm(p.grad.view(-1).to('cuda'), p = 2).item()
                if hf > h0:
                    self.g_dict[gidx][pidx] *= 2.
               

    def step(self, closure, model):
        """ Update parameters 
        """ 

        h0 = closure().item()

        grad_norm = self.get_grad_norm()
        self.update_params(closure, grad_norm)

        return h0



class KinAdgP(KinAdg):

    alias = 'P'

    def update_params(self, closure, grad_norm):
        for (gidx, group) in enumerate(self.param_groups):
            for (pidx, p) in enumerate(group['params']):
                if p.grad is None:
                    continue
                h0 = closure().item()
                step_size = np.sqrt(2.0 * h0 / self.g_dict[gidx][pidx]) # time 
                dp = -step_size * p.grad / grad_norm
                p.data.add_(dp)
                hf = closure().item()

                p_norm = torch.norm(p.grad.view(-1).to('cuda'), p = 2).item()
                if hf > h0:
                    self.g_dict[gidx][pidx] += p_norm ** 2



class KinOptg(KinAdg):

    alias = 'O'

    def update_params(self, closure, grad_norm):
        for (gidx, group) in enumerate(self.param_groups):
            for (pidx, p) in enumerate(group['params']):
                if p.grad is None:
                    continue
                h0 = closure().item()
                step_size = np.sqrt(2.0 * h0 / self.g_dict[gidx][pidx])
                dp = -step_size * p.grad / grad_norm
                p.data.add_(dp)
                hf = closure().item()
                if hf > h0:
                    p.data.add_(-dp)
                    orig_g = float(self.g_dict[gidx][pidx])
                    glist = []
                    for gstep in [-orig_g/10., orig_g/10.]:
                        curr_dl = hf - h0 
                        prev_dl = float('Inf')
                        g = float(orig_g)
                        i = 0
                        while curr_dl < prev_dl:
                            prev_dl = curr_dl
                            g += gstep 
                            if g == 0:
                                break
                            # get dl for gstep
                            _h0 = closure().item()
                            step_size = np.sqrt(2.0 * _h0 / g)
                            dp = -step_size * p.grad / grad_norm
                            p.data.add_(dp)
                            _hf = closure().item()
                            curr_dl = _hf - _h0
                            p.data.add_(-dp)
                            i += 1
                        # best g
                        best_g = g - gstep #/ gstep
                        glist.append((best_g, prev_dl))
                    # take all time best g 
                    g = min(glist, key = lambda x: x[1])[0]
                    # take step
                    h0 = closure().item()
                    step_size = np.sqrt(2.0 * h0 / g)
                    dp = -step_size * p.grad / grad_norm
                    p.data.add_(dp)
                    hf = closure().item()
                    dl = hf - h0
                    # save g 
                    self.g_dict[gidx][pidx] = g
                       