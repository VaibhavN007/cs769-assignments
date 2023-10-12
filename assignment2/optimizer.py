from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                
                # grad          : (30522, hidden_size = 768)
                # lr            : scalar
                # betas         : tuple(long, long)
                # eps           : scalar
                # weight_decay  : scalar
                # correct_bias  : scalar

                # State should be stored in this dictionary
                state = self.state[p]

                # initialize state if empty
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['t'] = torch.zeros([])

                # Access hyperparameters from the `group` dictionary
                lr, (beta1, beta2), eps = group['lr'], group['betas'], group['eps']
                
                # Update first and second moments of the gradients
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * grad**2
                state['t'] += 1

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                lr = lr * torch.sqrt(1 - beta2**state['t']) / (1 - beta1**state['t'])
                
                # Update parameters
                p.data = p.data - lr * state['m'] / (torch.sqrt(state['v']) + eps)

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                if group['weight_decay'] != 0.0:
                    p.data = p.data - lr * group['weight_decay'] * p.data

        return loss
