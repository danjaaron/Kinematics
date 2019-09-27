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

        v0 = 0.0

        # save model 
        torch.save(model.state_dict(), self.model_save_name)

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

        # drift on launch gradient
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

        # estimate land gradient
        new_grad_tens = torch.Tensor().to('cuda')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                else:
                    new_grad_tens = torch.cat((new_grad_tens, p.grad.view(-1).to('cuda')))
        new_grad_norm = torch.norm(new_grad_tens, p = 2)

        # get velocity at landing point 
        new_grad_dict = {}
        group_index = 0
        for group in self.param_groups:
            param_index = 0
            new_grad_dict[group_index] = {}
            for p in group['params']:
                if p.grad is None:
                    continue
                # store new grad 
                new_grad = p.grad/new_grad_norm 
                new_grad_dict[group_index][param_index] = new_grad
                param_index += 1
            group_index += 1
        '''

        # print("new h: {} old h: {} new/old: {}".format(new_h, old_h, new_h/old_h))
        # self.g += (new_h - old_h)/old_h
        # print("new g: {}".format(self.g))
        
            
        
        # self.g *= float(new_h)/float(old_h)
        

        """ Update g
        """ 
        # t_impact = math.sqrt(self.g/(2.*(abs(new_h - old_h))))
        
        #self.g = 2.*(old_h + old_h - new_h)/(t_impact**2)
        #self.past_g.append(self.g)
        # self.g = np.mean(self.past_g)
        # print(self.g)
        
        '''
        # self.g = (self.g + 2.*(old_h + old_h - new_h)/(t_impact**2))/2. #self.vf/t_impact
        
        

        # get new time 
        

        '''
        # adjust initial time of flight (lands at new_h, not 0)
        adjusted_vf = self.get_final_velocity(start_height = old_h + (old_h - new_h))#+ old_h - new_h)) # adjust height either (old_h + new_h) or (old_h + (old_h - new_h))
        t_impact = self.time_of_impact(adjusted_vf)
        print(t_impact)
        '''
        

        """ Estimate parabola vertex 
        """
        '''

        # restore model 
        model.load_state_dict(torch.load(self.model_save_name))
        
        # choose averaged initial velocity 
        group_index = 0
        for group in self.param_groups:
            param_index = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                # get new position 
                land_pos = new_location[group_index][param_index]
                # get original position 
                launch_pos = p.data.clone().detach()
                assert(land_pos.size() == launch_pos.size())
                same_count = 0
                diff_count = 0
                # TODO: Ensure pos tensors are always 2D, or adapt to higher dim
                
                
                """ Collision averaging / adjustment
                """
                #new_grad = new_grad_dict[group_index][param_index]
                old_grad = old_grad_dict[group_index][param_index] 
                # avg_grad = (new_grad + old_grad)/2.0 
                #avg_grad = (self.grad_dict[group_index][param_index] + p.grad)/2.0
                # update position 
                #p.data.add_(-t_impact, avg_grad)
                p.data = p.data + -t_impact*p.grad/old_grad_norm
                # p.data = (p.data + new_location[p])/2.0 
                # p.data = (p.data + (new_location[p] - p.data)/2.) #/2.
                
                param_index += 1
            group_index += 1

        final_h = closure().item()
        '''
        return loss
