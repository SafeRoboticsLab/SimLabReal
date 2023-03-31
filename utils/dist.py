import torch


def std2logvar(std):
    return 2 * torch.log(std)


def logvar2std(logvar):
    return torch.exp(logvar / 2)


def logvar2baselogvar(logvar, prior_std):
    return logvar - 2 * torch.log(prior_std)


def baselogvar2logvar(base_logvar, prior_std):
    return 2 * torch.log(prior_std) + base_logvar


def kl_inverse(q, c):
    import cvxpy as cvx
    import numpy as np
    '''Compute kl inverse using Relative Entropy Programming'''
    p_bernoulli = cvx.Variable(2)

    q_bernoulli = np.array([q, 1 - q])

    # print((q,c))

    constraints = [
        c >= cvx.sum(cvx.kl_div(q_bernoulli, p_bernoulli)),
        0 <= p_bernoulli[0], p_bernoulli[0] <= 1,
        p_bernoulli[1] == 1.0 - p_bernoulli[0]
    ]

    prob = cvx.Problem(cvx.Maximize(p_bernoulli[0]), constraints)

    # Solve problem
    prob.solve(verbose=False, solver=cvx.ECOS)  # solver=cvx.ECOS

    return p_bernoulli.value[0]
