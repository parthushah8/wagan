import itertools

import os
import json
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class RegularizedWassersteinDistance(nn.Module):
    """
    We find that it is important to ensure that λ is large enough,
    otherwise the projection of the image is excessively blurred.
    In addition to qualitative changes, smaller λ seems to make
    it harder to find Wasserstein adversarial examples, making
    the   radius go up as λ gets smaller. In fact, for λ = (1, 10)
    and almost all of λ = 100, the blurring is so severe that no
    adversarial example can be found.

    In contrast, we find that increasing p for the Wasserstein
    distance used in the cost matrix C seems to make the images
    more “blocky”. Specifically, as p gets higher tested, more
    pixels seem to be moved in larger amounts. This seems to
    counteract the blurring observed for low λ to some degree.
    Naturally, the   radius also grows since the overall cost of
    the transport plan has gone up.

    epsilon = 1 / lambda
    """

    def __init__(self, d, p = 2, epsilon = 0.2, max_iter = 50, tolerance = 1e-6, cost_function = None, device = "cpu"):
        super(RegularizedWassersteinDistance, self).__init__()
        self.d = d
        self.n = d * d
        self.p = p
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.device = device

        self.cost = torch.zeros(self.n, self.n)
        for i, j, k, l in itertools.product(range(self.d), repeat = 4):
            self.cost[i * self.d + j, k * self.d + l] = ((i - k) ** 2 + (j - l) ** 2) ** (self.p / 2) if(cost_function is None) else cost_function((i, j), (k, l))

        self.kernel = torch.exp(-self.cost / self.epsilon) 

        self.cost = self.cost.to(self.device)
        self.kernel = self.kernel.to(self.device)

    def forward(self, x, y):
        b = x.shape[0]

        x = x.reshape(b, -1).to(self.device)
        y = y.reshape(b, -1).to(self.device)

        u = torch.ones((b, self.n)).to(self.device) / self.n
        v = torch.ones((b, self.n)).to(self.device) / self.n

        for _ in range(self.max_iter):
            pu, pv = u, v
          
            u = torch.div(x, v @ self.kernel.T + 1e-8)
            v = torch.div(y, u @ self.kernel + 1e-8) 

            if(((u - pu).abs() + (v - pv).abs()).sum(1).mean() < self.tolerance):
                break
        
        pi = (u.diag_embed() @ self.kernel @ v.diag_embed()) 
        distance = torch.sum(pi * self.cost)
        
        return distance