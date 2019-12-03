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
        self.last_grad_norm = None 
        self.last_loss = None 
        default_dict = {'g': float(self.g)}
        # print(super(self.__class__)
        super(Kin, self).__init__(params, defaults = default_dict)

    def __setstate__(self, state):
        super(Kin, self).__setstate__(state)

    def get_grad_norm(self):
        """ Get norm of gradient
        """
        generator = [k.grad.view(-1).cuda() for p in ( g['params'] for g in self.param_groups ) for k in p if k.grad is not None]
        grad_norm = torch.norm(torch.cat(generator), p = 2).item()
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

    def step(self, closure):
        """ Update parameters 
        """ 

        self.closure = closure

        h0 = closure().item()

        grad_norm = self.get_grad_norm()
        step_size = self.get_step_size(h0)
        self.grad_norm = grad_norm
        self.loss = h0 
        self.update_params(step_size, grad_norm)

        hf = closure().item()

        self.update_g(h0, hf)

        self.last_grad_norm = grad_norm
        self.last_loss = h0 

        return h0

class KinVel(Optimizer):

    alias = 'V'

    def __init__(self, params):
        self.g = 1.0
        self.v0 = 0.
        self.last_grad_norm = None 
        self.last_loss = None 
        default_dict = {'g': float(self.g)}
        # print(super(self.__class__)
        super(KinVel, self).__init__(params, defaults = default_dict)

    def __setstate__(self, state):
        super(KinVel, self).__setstate__(state)

    def get_grad_norm(self):
        """ Get norm of gradient with velocity included
        """
        '''
        generator = [k.grad.view(-1).to('cuda') for p in ( g['params'] for g in self.param_groups ) for k in p if k.grad is not None]
        # OLD:
        generator.append(torch.Tensor([self.v0]).to('cuda'))
        # above is not in test, but should be technically correct
        grad_norm = torch.norm(torch.cat(generator), p = 2).item()
        return grad_norm
        '''
        grad_sum = 0.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_sum += (p.grad ** 2).sum()
        grad_sum += self.v0 ** 2
        self.grad_norm = math.sqrt(grad_sum)
        return self.grad_norm


    def update_params(self, vf, grad_norm, undo = False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # OLD:
                # t = self.loss / vf 
                if undo:
                    self.t *= -1.0
                p.data.add_(-self.t, p.grad / grad_norm) #* torch.norm(p.grad, p=2).item() / grad_norm)

    def get_vf(self, v0, a, dx):
        return math.sqrt(v0**2 + 2.0*a*dx)

    def update_g(self, h0, hf):
        if h0 > hf:
            self.vf = self.get_vf(self.v0, self.g, abs(h0 - hf))
        else:
            # OLD: self.g += (hf / vf)**2 
            self.g += (abs(hf) / (self.t**2))
            self.update_params(self.vf, self.grad_norm, undo = True)
            self.vf = 0.
            

        '''
        # update v 
        # self.vf = np.sqrt(self.v0**2 + 2.0*self.g*(h0 - hf))
        if hf > h0:
            # self.g *= 2.0 #+= ((hf - h0) / ((h0 / self.vf) ** 2))
            self.g += (hf / self.vf) ** 2
            self.update_params(self.vf, self.grad_norm, True)
            self.vf = 0.
        else:
            # self.g = ((h0 - hf) / self.vf) ** 2
            t = h0 / self.vf
            self.vf = (h0 - hf) / t
        '''

    def step(self, closure):
        """ Update parameters 
        """ 

        self.closure = closure

        h0 = closure().item()
        self.loss = h0 
        vf = self.get_vf(self.v0, self.g, h0)
        self.vf = vf

        # OLD: self.t = h0 / vf 
        self.t = 2.0*h0 / (self.v0 + self.vf)



        grad_norm = self.get_grad_norm()
        self.grad_norm = grad_norm

        self.update_params(vf, grad_norm)

        hf = closure().item()

        self.update_g(h0, hf)

        self.v0 = self.vf

        return h0

class KinOsc(Kin):

    alias = 'O'

    # if oscillating, will be closer to earlier positions than later ones

    def __init__(self, params):
        super(KinOsc, self).__init__(params)
        # track last track_p params
        self.last_p = list()
        self.oscillating = False
        self.track_p = 5

    def get_grad_norm(self):
        """ Get norm of gradient
        """
        flat_grad = torch.Tensor().to('cuda')
        flat_params = torch.Tensor().to('cuda') # for distances
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                else:
                    flat_grad = torch.cat((flat_grad, p.grad.view(-1).to('cuda')))
                    flat_params = torch.cat((flat_params, p.data.view(-1).to('cuda')))
        grad_norm = torch.norm(flat_grad, p = 2)
        # check distance between last 3 
        self.oscillating = False
        if 2 < len(self.last_p) <= self.track_p:
            # get current param distance from past params
            distances = [torch.dist(flat_params, _).item() for _ in self.last_p]
            # check if reverse monotonic distances
            self.oscillating = not all(distances[i] >= distances[i+1] for i in range(len(distances) - 1))
            print("OSC: ", self.oscillating)
            print("dist: ", distances)
            if self.oscillating:
                # empty buffer if oscillating 
                self.last_p = list()
            elif len(self.last_p) == self.track_p:
                # roll tracking buffer
                self.last_p = np.roll(self.last_p, -1)
                self.last_p[-1] = flat_params
        else:
            self.last_p.append(flat_params)
        return grad_norm

    def get_step_size(self, loss):
        ss = np.sqrt(2.0*loss/self.g)
        if self.oscillating:
            correct_ss = ss / 2.0
            self.g = 2.0 * loss / (correct_ss ** 2)
            ss = correct_ss
        return ss




class KinE(Kin):

    alias = 'E'

    def update_params(self, step_size, grad_norm):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(-step_size, p.grad / grad_norm)

    def update_g(self, h0, hf):
        if self.last_grad_norm is None:
            print("last grad is none")
            # if hf > h0:
                # self.g *= 2.0   
        else: 
            print("last grad NOT none")
            # conserve energy s
            new_g = (self.g * self.last_loss + 0.5 * (self.last_grad_norm ** 2 - self.grad_norm ** 2)) / h0
            new_g = new_g.item()
            print("new ", new_g)
            if new_g > 0:
                self.g = new_g
            elif hf > h0:
                self.g *= 2.0
            '''
            if new_g.item() > 0:
                print("new ", new_g)
                self.g = abs(new_g.item())
            elif hf > h0:
                self.g *= 2.
            '''





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


'''
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
'''