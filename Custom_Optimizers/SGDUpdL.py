import torch
from torch.optim import Optimizer
from typing import List, Optional
from torch import Tensor
from .Lupdate import LUPDATE
import functools
import matplotlib.pyplot as plt
import numpy as np

#### This is used as a decorator (I am not quite sure what this is) on the step() method in the custom optimizer
#### I copy pasted it from PyTorch as it is needed to compute and track the gradients in the background, which is the basic functionality of PyTorch
def _use_grad_for_differentiable(func):
    def _use_grad(self, *args, **kwargs):
        import torch._dynamo
        prev_grad = torch.is_grad_enabled()
        try:
            # Note on graph break below:
            # we need to graph break to ensure that aot respects the no_grad annotation.
            # This is important for perf because without this, functionalization will generate an epilogue
            # which updates the mutated parameters of the optimizer which is *not* visible to inductor, as a result,
            # inductor will allocate for every parameter in the model, which is horrible.
            # With this, aot correctly sees that this is an inference graph, and functionalization will generate
            # an epilogue which is appended to the graph, which *is* visible to inductor, as a result, inductor sees that
            # step is in place and is able to avoid the extra allocation.
            # In the future, we will either 1) continue to graph break on backward, so this graph break does not matter
            # or 2) have a fully fused forward and backward graph, which will have no_grad by default, and we can remove this
            # graph break to allow the fully fused fwd-bwd-optimizer graph to be compiled.
            # see https://github.com/pytorch/pytorch/issues/104053
            torch.set_grad_enabled(self.defaults['differentiable'])
            torch._dynamo.graph_break()
            ret = func(self, *args, **kwargs)
        finally:
            torch._dynamo.graph_break()
            torch.set_grad_enabled(prev_grad)
        return ret
    functools.update_wrapper(_use_grad, func)
    return _use_grad

def weight_update(params: List[Tensor],
        grad_list: List[Tensor],
        step_sizes: List[Tensor],
        weight_decay: float,
        momentum: float,
        momentum_buffer_list: List[Optional[Tensor]]):
        for i, param in enumerate(params):
            step_size = step_sizes[i]
            grad = grad_list[i]

            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)

            if momentum != 0:
                buf = momentum_buffer_list[i]
                if buf is None:
                    buf = torch.clone(grad).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(grad, alpha=1)  # Dampening = 0

            param.addcmul_(grad, step_size, value=-1)  # Update weights using individual learning rates and sign of gradient

def lr_update(params: List[Tensor],
    weights1: Tensor,
    weights2: Tensor,
    weights3: Tensor,
    step_sizes: List[Tensor],
    step_size_min: float,
    step_size_max: float,
    etaminus: float,
    etaplus: float,
    differentiable: bool=False,):
    s1 = step_sizes
    s2 = s1[0]
    s3 = s2[0]
    #print(len(s1))
    #print(len(s2))
    #print(len(s3))
    for i, (param, w1, w2, w3) in enumerate(zip(params, weights1, weights2, weights3)):
        if param.grad is None:
            continue
        step_size = step_sizes[i]
        #print('Step',i, step_size)
        dw_epochA = w2.data-w1.data  #  Calculate first differene in weights
        dw_epochB = w3.data-w2.data   # Calculate second difference in weights
        if differentiable:
            signs = dw_epochA.mul(dw_epochB.clone()).sign()
        else:
            signs = dw_epochA.mul(dw_epochB).sign()
        
        signs[signs.gt(0)] = etaplus     
        signs[signs.lt(0)] = etaminus
        signs[signs.eq(0)] = 1               
        step_size.mul_(signs).clamp_(step_size_min, step_size_max)  
    
        ### for dir<0, dfdx=0
        ### for dir>=0 dfdx=dfdx
        restore = torch.zeros_like(signs, requires_grad=False)
        restore[signs.eq(etaminus)] = 1                           # tracks which weights had gradient set to zero --> 1 otherwise 0
        param.data.addcmul_(dw_epochB.detach(), restore, value=-1)    # reverts tracked weights back to w2
    #print('hey', i)

def get_lr_stats(step_sizes: List[Tensor], lr_mean: List, lr_std: List, lr_all: List, track_lr: bool=False,):
    if track_lr:
        single_tensor = torch.cat([x.flatten() for x in step_sizes])
        lr_mean.append(single_tensor.mean().item())
        lr_std.append(single_tensor.std().item())
        float_list = [t.item() for t in single_tensor]
        lr_all.append(list(float_list))
    else:
        pass

def Clone_Parameters(model_parameters):
    Param_list = []
    for p in model_parameters:
        Param_list.append(p.detach().clone())
    return Param_list

#### Define the custom optimizer
class SGDUPDL(Optimizer):
    def __init__(self, params, M=1, L=1, lr=1e-2, etas=(0.5, 1.2), lr_limits=(1e-6, 50), momentum=0, \
                 weight_decay=0, *, track_lr: bool = False, differentiable: bool = False,L_update_type = 'linear', L_update_step = 'round up', max_L = 50000, min_L = 0, s_plus = 1, s_min = -1):
            if not 0.0 <= lr:
                raise ValueError(f"Invalid learning rate: {lr}")
            if momentum < 0.0:
                raise ValueError(f"Invalid momentum value: {momentum}")
            if not 0.0 < etas[0] < 1.0 < etas[1]:
                raise ValueError(f"Invalid eta values: {etas[0]}, {etas[1]}")
            if not M!=L:
                raise ValueError(f"For M=L, use Rprop")
            if not L%M==0:
                raise ValueError(f"L={L} must be integer multiple of M={M}")
        #### Make hyperparameters accessible through a dictionary
            defaults = dict(M=M, L=L, lr=lr, etas=etas, lr_limits=lr_limits, momentum=momentum,
                            weight_decay=weight_decay, differentiable=differentiable, track_lr=track_lr)
        #### super() makes the class inherit properties from PyTorch's Optimizer class
            super().__init__(params, defaults)
        #### Giving the class attributes that can be accessed later to update learning rates
            self.step_sizes = []  # Initialize step_sizes attribute
            self.lr_counter = 0  # Counts learning rate updates
            self.data_tally = 0  # Counts volume of data fed through
            self.weights1 = []  # Used in step-size update
            self.weights2 = []  # Used in step-size update
            self.weights3 = []  # Used in step-size update
            self.lr_mean, self.lr_std = [], []
            self.L_sizes = []
            self.L = L
            self.M = M
            self.L_continues = L
            self.next_L = [0]
            self.L_update_type = L_update_type #Which function to use for L updates
            self.L_update_step = L_update_step
            self.max_L = max_L #The maximum length of L
            self.min_L = min_L
            self.s_plus = s_plus
            self.lr_all = []
            if s_min == -1:
                self.s_min = s_plus
            else:
                self.s_min = s_min
        
    #### Needed for the decorator
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("differentiable", False)
    
    def L_update(self,M):
        func = LUPDATE(L = self.L, max_L = self.max_L, min_L = self.min_L, M = M, s_plus = self.s_plus,  
                       s_min = self.s_min, 
                       lr_mean = self.lr_mean, L_update_type = self.L_update_type, L_update_step = self.L_update_step, 
                        L_continues = self.L_continues)
        L, self.L_continues = func.L_update()
        return L

    @_use_grad_for_differentiable ### This line is the decorator
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        params_with_grad = []
        grad_list = []
        momentum_buffer_list = []

        #### PyTorch has in-built parameter groups which allow you to change hyperparameters for different layers
        #### This implementation only handles models with one parameter group! (Eg. can't change learning rate by layer)
        
        for group in self.param_groups:
            for p in group['params']:       #### Iterating through layers
                if p.grad is None:          #### .grad calculates gradient
                    continue
                params_with_grad.append(p)
                grad = p.grad

                if p.grad.is_sparse:
                    raise RuntimeError("Rprop does not support sparse gradients")#?
                    
                grad_list.append(grad)
                state = self.state[p]
                momentum_buffer_list.append(state.get('momentum_buffer'))
                
                if len(state)==0:       # First time optimizerz is called initialize internal state and track steps
                    state["step"] = 0
                    state["step_size"]  = (grad.new().resize_as_(grad).fill_(group["lr"]))
                    self.step_sizes.append(state["step_size"])

                state["step"] += 1
            
            #L, M = group["L"], group["M"]  # Use L and M hyperparameters
            M = group["M"]  # Use L and M hyperparameters
            momentum, weight_decay = group["momentum"], group["weight_decay"]
            
            if self.data_tally % self.L == 0 and self.lr_counter == 0:  # First iteration of lr-update on first call
                self.weights1 = Clone_Parameters(group["params"])   # Save network parameters
                get_lr_stats(self.step_sizes, self.lr_mean, self.lr_std, self.lr_all, group["track_lr"])
                self.lr_counter += 1
                self.L_sizes.append(self.L)
                self.next_L.append(self.L + self.data_tally)
                #print(self.data_tally, self.L, self.next_L[-1], self.lr_counter)
            
            
            weight_update(params_with_grad, grad_list, self.step_sizes, weight_decay, momentum, momentum_buffer_list)  # Update weights, everytime
            self.data_tally += M   # Add weight mini-batch size to data seen, everytime
            
            #if self.data_tally % L == 0 and self.lr_counter == 1:  # Second iteration of lr-update
            if self.data_tally % self.next_L[-1] == 0 and self.lr_counter == 1:  # Second iteration of lr-update
                self.weights2 = Clone_Parameters(group["params"])   # Save network parameters
                get_lr_stats(self.step_sizes, self.lr_mean, self.lr_std, self.lr_all, group["track_lr"])
                self.lr_counter += 1
                self.L_sizes.append(self.L)
                self.L = self.L_update(M)
                self.next_L.append(self.L + self.data_tally)
                #print(self.data_tally, self.L, self.next_L[-1], self.lr_counter)
                
            #elif self.data_tally % L == 0 and self.lr_counter >= 2: # Third iteration of lr-update
            elif self.data_tally % self.next_L[-1] == 0 and self.lr_counter >= 2: # Third iteration of lr-update
                etaminus, etaplus = group["etas"]
                step_size_min, step_size_max = group["lr_limits"]
                self.weights3 = Clone_Parameters(group["params"])   # Save network parameters
                if len(self.step_sizes) < len(group["params"]):
                    min_lr = group["lr_limits"][0]  # Define min_lr here using the lr_limits
                    while len(self.step_sizes) < len(group["params"]):
                        self.step_sizes.append(torch.full_like(group["params"][len(self.step_sizes)], min_lr))
                lr_update(group["params"], self.weights1, self.weights2, self.weights3, self.step_sizes, \
                                 step_size_min, step_size_max, etaminus, etaplus)
                get_lr_stats(self.step_sizes, self.lr_mean, self.lr_std, self.lr_all, group["track_lr"]) # Tracking lr updates for debugging
                self.weights3 = Clone_Parameters(group["params"])
                for i in range(len(self.weights1)):
                    self.weights1[i] = self.weights2[i].detach().clone()
                    self.weights2[i] = self.weights3[i].detach().clone()
                self.lr_counter += 1
                self.L_sizes.append(self.L)
                self.L = self.L_update(M)
                self.next_L.append(self.L + self.data_tally)
                #print(self.data_tally, self.L, self.next_L[-1], self.lr_counter)

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
                    
        return loss

