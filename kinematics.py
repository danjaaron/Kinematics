import math

import numpy as np
import torch
from torch.optim import Optimizer

class Kin(Optimizer):
    alias = 'K'

    def __init__(self, params):
        self.g = 1.0
        self.last_grad_norm = None
        self.last_loss = None
        default_dict = {'g': float(self.g)}
        # print(super(self.__class__)
        super(Kin, self).__init__(params, defaults=default_dict)
        self.prev_pdata = dict()

    def __setstate__(self, state):
        super(Kin, self).__setstate__(state)

    def get_grad_norm(self):
        """ Get norm of gradient
        """
        generator = [k.grad.view(-1).cuda() for p in (g['params'] for g in self.param_groups) for k in p if
                     k.grad is not None]
        grad_norm = torch.norm(torch.cat(generator), p=2).item()
        return grad_norm

    def get_step_size(self, loss):
        return np.sqrt(2.0 * loss / self.g)

    def update_params(self, step_size, grad_norm):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.prev_pdata[p] = (p.data, p.grad, grad_norm, step_size)
                p.data.add_(-step_size, p.grad / grad_norm)

    def update_g(self, h0, hf):
        if hf > h0:
            self.g *= 2.0

    def step(self, closure):
        """ Update parameters
        """

        self.closure = closure

        h0 = closure().item()
        self.h0 = h0

        grad_norm = self.get_grad_norm()
        step_size = self.get_step_size(h0)
        self.grad_norm = grad_norm
        self.loss = h0
        self.update_params(step_size, grad_norm)

        hf = closure().item()
        self.hf = hf

        self.update_g(h0, hf)

        self.last_grad_norm = grad_norm
        self.last_loss = h0

        return h0

class Bouncy(Optimizer):
    alias = 'B'

    """
    Has an independent height, separate from loss, and an independent vertical velocity. 

    If its height <= loss, it bounces (collides with loss landscape), and if not it continues moving. 
    """

    def __init__(self, params):
        super(Bouncy, self).__init__(params, defaults=dict())
        self.g = -9.8
        self.alpha = 1e-2 # step size - units of time 
        self.h = None # initial height - get from closure
        self.h_vel = 0.0 # arbitrary starting initial upward velocity 
        # self.dir = 1.0 
        self.position = {}
        self.velocity = {}
        

    def bounce(self):
        """
        Update velocity from collision with loss landscape
        """
        # if self.h_vel < 0.:
            # self.h_vel = -0.8*(self.h_vel)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # reflect the velocity vector
                # NOTE: treating p.grad as the normal of landscape wall
                # normal = p.grad 
                # print(self.velocity[p])
                # print(normal)
                # print(torch.mm(self.velocity[p], normal))
                new_v = (p.grad + self.velocity[p])/2.0
                p.data.add_((new_v.pow(2) - self.velocity[p].pow(2))/(2.*self.g))
                self.velocity[p] = new_v #p.grad #(self.velocity[p] + p.grad)/2.
                # self.velocity[p] = self.velocity[p] - 2.0*torch.mm(self.velocity[p], normal)*normal
                #p.data.add_(self.alpha*self.velocity[p])
                self.position[p] = p.data

    def drift(self):
        """
        Update position, but not velocity 
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                #self.h += -self.alpha*self.velocity[p].sum()
                p.data.add_(-self.alpha, self.velocity[p])
                self.position[p] = p.data

    def water(self):
        """
        Update position and velocity
        """
        print('water')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.velocity[p].add_(p.grad).div_(2.)
                delta = -self.alpha*self.velocity[p]
                p.data.add_(delta)
                # self.h += delta
                self.position[p] = p.data

    def air(self):
        """
        Update position, but not velocity 
        """
        print('air')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                #self.h += -self.alpha*self.velocity[p].sum()
                p.data.add_(-self.alpha, self.velocity[p])
                self.position[p] = p.data

    def step(self, closure):
        if self.h is None:
            self.h = closure()*1.
            # get initial positions and velocities 
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    self.position[p] = p.data 
                    self.velocity[p] = p.grad
        # update verticals
        # self.h_vel += self.alpha*self.g 
        # print("ADDING: ", self.alpha*self.g, self.h_vel)
        # self.h += self.h_vel*self.alpha + 0.5*self.g*(self.alpha**2)
        # check if collision 
        sea_level = closure()
        h_diff = self.h - sea_level
        print('h {0}, h_vel {1}'.format(self.h, self.h_vel))
        if h_diff < 0:
            self.water()
            self.h_vel += self.alpha*abs(self.g)
            self.h += self.alpha*self.h_vel + 0.5*abs(self.g)*(self.alpha**2)
        else:
            self.air()
            self.h_vel += self.alpha*self.g
            self.h += self.alpha*self.h_vel + 0.5*self.g*(self.alpha**2)
        '''
        self.h_vel np.sign(h_diff))/2.0
        self.h_vel += self.g*self.alpha*(int(self.h < sea_level))
        self.h += self.h_vel*self.alpha + 0.5*self.g*(self.alpha**2)*(int(self.h < sea_level))
        print('H > sea: ', self.h > sea_level)
        print('h {0:.2f} v {1:.2f}'.format(self.h, self.h_vel))
        # self.bounce()
        if self.h > sea_level:
            self.drift()
        else:
            self.bounce()
        '''
        

        '''
        if self.h <= landscape_height:
            print('bounce!', self.h - landscape_height, self.h_vel)
            self.g = abs(self.g)
            self.bounce()
        else:
            self.g = -abs(self.g)
            print('drift ', self.h - landscape_height, self.h_vel)
            self.drift()
        '''
        # print(self.h, landscape_height, self.h_vel)

class Time(Optimizer):
    alias = 'T'

    def __init__(self, params):
        super(Time, self).__init__(params, defaults=dict())
        self.g = dict()
        self.v = dict() #
        self.last_step = dict()
        self.default_g = 10.

    def update_params(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if not p in self.g:
                    self.g[p] = self.default_g
                    self.v[p] = 0. #
                    self.last_step[p] = None



                v0 = self.v[p] #

                a = self.g[p]
                d = self.closure().item()

                vf = math.sqrt(v0*v0 + 2.*a*d)

                #t = max(0., (vf - v0)/a)
                t = (vf - v0)/a

                #full_norm = math.sqrt(p.grad.norm()**2 + v0**2)
                full_norm = p.grad.norm()
                step_size = -t*p.grad/full_norm # p.grad.norm() # self.g[p]

                # gradient descent over your own hyperparams?

                """
                if self.last_step[p] is None:
                    self.last_step[p] = step_size
                ss = (step_size + self.last_step[p])/2.0
                self.last_step[p] = ss
                """
                ss = step_size
                p.data.add_(ss)

                hf = self.closure().item()

                #print('height: ', d, hf)
                if hf >= d:
                    p.data.add_(-ss)
                    #self.g[p] *= 2.
                    self.g[p] += (vf**2 - v0**2)/(2.*(hf))
                else:
                    self.v[p] = vf 
                    #self.g[p] /= 1.1




    def step(self, closure):
        self.closure = closure
        #self.h0 = closure.item()
        self.update_params()
        #self.hf = closure.item()
        #if self.hf > self.h0:
        #    self.update_g()
        return self.closure().item()

"""
class Relaunch(Kin):
    
    alias='R'

    def __init__(self, params):
        super(Relaunch, self).__init__(params)

    '''
    def update_params(self, step_size, grad_norm):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.prev_pdata[p] = (step_size, p.grad)
                p.data.add_(-step_size, p.grad)
    '''

    def update_g(self, h0, hf):
        if hf > h0:
            #self.g *= 2
            self.g *= 
            # undo last change 
            self.undo_step()

    def undo_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(self.prev_pdata[p][0], self.prev_pdata[p][1]) #, self.prev_pdata[p][1] / self.prev_pdata[p][2])
        print('undid step')
"""

class KinConvert(Kin):
    ''' Energy conversion implementation
    '''
    alias = 'C'

    def update_g(self, h0, hf):
        pass

    def potential_energy(self):
        ''' Returns potential energy
        '''


    def kinetic_energy(self):
        ''' Returns kinetic energy
        '''

    def energy(self):
        ''' Returns total energy
        '''
        k = self.kinetic_energy()
        p = self.potential_energy()
        self.energy = float(k + p)

    def step(self, closure):
        # initial energy
        self.closure = closure
        self.energy()

        # step
        super().step(self, closure)

        # final energy
        self.energy()

        # convert energy to potential

class KinVel(Optimizer):
    alias = 'V'

    def __init__(self, params):
        self.g = 1.0
        self.v0 = 0.
        self.beta = 0.9
        self.last_grad_norm = None
        self.last_loss = None
        default_dict = {'g': float(self.g)}
        self.mdict = {}
        # print(super(self.__class__)
        super(KinVel, self).__init__(params, defaults=default_dict)

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
        if not self.mdict:
            for gidx, group in enumerate(self.param_groups):
                self.mdict[gidx] = dict()
                for pidx, p in enumerate(group['params']):
                    self.mdict[gidx][pidx] = 0.

        grad_sum = 0.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_sum += (p.grad ** 2).sum()
        # grad_sum += self.v0 ** 2
        self.grad_norm = math.sqrt(grad_sum)
        return self.grad_norm

    def update_params(self, vf, grad_norm, undo=False):
        for gidx, group in enumerate(self.param_groups):
            for pidx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                # OLD:
                # t = self.loss / vf 
                # if undo:
                # self.t *= -1.0
                if undo:
                    p.data.add_(-self.mdict[gidx][pidx])
                else:
                    self.mdict[gidx][pidx] = (1 - self.beta) * self.mdict[gidx][pidx] + self.beta * (
                                -self.t * p.grad / grad_norm)
                    p.data.add_(self.mdict[gidx][pidx])
                # p.data.add_(-self.t, p.grad / grad_norm) #* torch.norm(p.grad, p=2).item() / grad_norm)

    def get_vf(self, v0, a, dx):
        return math.sqrt(v0 ** 2 + 2.0 * a * dx)

    def update_g(self, h0, hf):

        if h0 > hf:
            # correct vf 
            self.vf = self.get_vf(self.v0, self.g, abs(h0 - hf))
        elif hf > h0:
            self.g += (hf / self.vf) ** 2
            self.vf = self.get_vf(self.v0, self.g, abs(h0 - hf))
            self.update_params(self.vf, self.grad_norm, undo=True)

        '''
        if h0 > hf:
            self.vf = self.get_vf(self.v0, self.g, abs(h0 - hf))
        else:
            # OLD: self.g += (hf / vf)**2 
            self.g += (abs(hf) / (self.t**2))
            self.update_params(self.vf, self.grad_norm, undo = True)
            self.vf = 0.
        '''

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
        self.t = 2.0 * h0 / (self.v0 + self.vf)

        grad_norm = self.get_grad_norm()
        self.grad_norm = grad_norm

        self.update_params(vf, grad_norm)

        hf = closure().item()

        # self.vf = self.get_vf(self.v0, g, abs(h0 - hf))

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
        flat_params = torch.Tensor().to('cuda')  # for distances
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                else:
                    flat_grad = torch.cat((flat_grad, p.grad.view(-1).to('cuda')))
                    flat_params = torch.cat((flat_params, p.data.view(-1).to('cuda')))
        grad_norm = torch.norm(flat_grad, p=2)
        # check distance between last 3 
        self.oscillating = False
        if 2 < len(self.last_p) <= self.track_p:
            # get current param distance from past params
            distances = [torch.dist(flat_params, _).item() for _ in self.last_p]
            # check if reverse monotonic distances
            self.oscillating = not all(distances[i] >= distances[i + 1] for i in range(len(distances) - 1))
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
        ss = np.sqrt(2.0 * loss / self.g)
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

                    t_flight = math.sqrt(2.0 * h0 / group['g'])

                    p.data.add_(-t_flight * group['m1'])

                    group['m1'] = (1 - self.b1) * group['m1'] + self.b1 * (p.grad / grad_norm)

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
                step_size = np.sqrt(2.0 * h0 / self.g_dict[gidx][pidx])  # time
                dp = -step_size * p.grad / grad_norm
                p.data.add_(dp)
                hf = closure().item()

                p_norm = torch.norm(p.grad.view(-1).to('cuda'), p=2).item()
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
                step_size = np.sqrt(2.0 * h0 / self.g_dict[gidx][pidx])  # time
                dp = -step_size * p.grad / grad_norm
                p.data.add_(dp)
                hf = closure().item()

                p_norm = torch.norm(p.grad.view(-1).to('cuda'), p=2).item()
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
