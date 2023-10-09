import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from docplex.mp.model import Model
import seaborn as sns; sns.set()

np.random.seed(10)

n=3
action_spaces = np.array([3,4,5])
reward = np.random.uniform(0, 1, size=(action_spaces))

initial_pi = ([np.random.uniform(0,1,size=int(i)) for i in (action_spaces)])
for pi in initial_pi:
    pi/=np.sum(pi)
eta = 0.1
tau = 0
pi_star = np.copy(initial_pi)
new_pi_star = np.copy(initial_pi)

pi = np.copy(initial_pi)
theta = np.copy(initial_pi)
for agent in range(n):
    theta[agent] = np.log(theta[agent])

NE_gaps = []
N = 10000
for i in tqdm(range(N)):
    NE_gap = []
    next_pi = np.copy(pi)
    next_theta = np.copy(theta)
    for agent in range(n):
        r = np.copy(reward)
        r = np.transpose(r)
        for left_agent in range(agent):
            r = np.matmul((r), pi[left_agent])
        r = np.transpose(r)
        for right_agent in range(n-1,agent, -1):
            r = np.matmul(r, pi[right_agent])
        r = r-np.dot(r,pi[agent])
        NE_gap.append(np.max(r))
        
        next_theta[agent] += eta*r   # this is for NPG
        next_pi[agent] = np.exp(next_theta[agent])        
        next_pi[agent]/=np.sum(next_pi[agent])
    NE_gaps.append(NE_gap)
    pi = np.copy(next_pi)
    theta = np.copy(next_theta)
NE_gaps = np.array(NE_gaps)
NPG_NE_gaps = np.sum(NE_gaps, axis=1)
# plt.plot(NPG_NE_gaps)
# plt.show()

pi = np.copy(initial_pi)
theta = np.copy(initial_pi)
for agent in range(n):
    theta[agent] = np.log(theta[agent])

NE_gaps = []
for i in tqdm(range(N)):
    NE_gap = []
    next_pi = np.copy(pi)
    next_theta = np.copy(theta)
    for agent in range(n):
        r = np.copy(reward)
        r = np.transpose(r)
        for left_agent in range(agent):
            r = np.matmul((r), pi[left_agent])
        r = np.transpose(r)
        for right_agent in range(n-1,agent, -1):
            r = np.matmul(r, pi[right_agent])
        r = r-np.dot(r,pi[agent])
        NE_gap.append(np.max(r))
        next_theta[agent] += eta*r*pi[agent]
        next_pi[agent] = np.exp(next_theta[agent])

        next_pi[agent]/=np.sum(next_pi[agent])
    NE_gaps.append(NE_gap)
    pi = np.copy(next_pi)
    theta = np.copy(next_theta)
NE_gaps = np.array(NE_gaps)
PG_NE_gaps = np.sum(NE_gaps, axis=1)
# plt.plot(PG_NE_gaps)
eta = 0.1
tau = 0.05
pi_star = np.copy(initial_pi)
new_pi_star = np.copy(initial_pi)

pi = np.copy(initial_pi)
theta = np.copy(initial_pi)
for agent in range(n):
    theta[agent] = np.log(theta[agent])

NE_gaps = []
for i in tqdm(range(N)):
    NE_gap = []
    next_pi = np.copy(pi)
    next_theta = np.copy(theta)
    for agent in range(n):
#         r = np.zeros(a)
        r = np.copy(reward)
        r = np.transpose(r)
        for left_agent in range(agent):
            r = np.matmul((r), pi[left_agent])
        r = np.transpose(r)
        for right_agent in range(n-1,agent, -1):
            r = np.matmul(r, pi[right_agent])
        r = r-np.dot(r,pi[agent])
        NE_gap.append(np.max(r))
        
        next_pi[agent] = np.power(pi[agent],(1-eta*tau))*np.exp(eta*r)  #NPG+regularization
        next_pi[agent]/=np.sum(next_pi[agent])
    NE_gaps.append(NE_gap)
    pi = np.copy(next_pi)
    theta = np.copy(next_theta)

NE_gaps = np.array(NE_gaps)
NPG_regularization_NE_gaps = np.sum(NE_gaps, axis=1)
# plt.plot(NPG_regularization_NE_gaps)

eta = 0.1
tau = 0.005
pi_star = np.copy(initial_pi)
new_pi_star = np.copy(initial_pi)

pi = np.copy(initial_pi)
theta = np.copy(initial_pi)
for agent in range(n):
    theta[agent] = np.log(theta[agent])

NE_gaps = []
# N = 2000
for i in tqdm(range(N)):
    NE_gap = []
    next_pi = np.copy(pi)
    next_theta = np.copy(theta)
    for agent in range(n):
#         r = np.zeros(a)
        r = np.copy(reward)
        r = np.transpose(r)
        for left_agent in range(agent):
            r = np.matmul((r), pi[left_agent])
        r = np.transpose(r)
        for right_agent in range(n-1,agent, -1):
            r = np.matmul(r, pi[right_agent])
        r = r-np.dot(r,pi[agent])
        NE_gap.append(np.max(r))
        
        # next_theta[agent] += eta*r   # this is for NPG
        next_theta[agent] += eta*(r*pi[agent] + tau*(1-pi[agent]*action_spaces[agent]))
        next_pi[agent] = np.exp(next_theta[agent])        
        next_pi[agent]/=np.sum(next_pi[agent])
    NE_gaps.append(NE_gap)
    pi = np.copy(next_pi)
    theta = np.copy(next_theta)

NE_gaps = np.array(NE_gaps)
PG_log_barrier_NE_gaps = np.sum(NE_gaps, axis=1)
# plt.plot(PG_log_barrier_NE_gaps)

eta = 0.1
tau = 0.005
pi_star = np.copy(initial_pi)
new_pi_star = np.copy(initial_pi)

pi = np.copy(initial_pi)
theta = np.copy(initial_pi)
for agent in range(n):
    theta[agent] = np.log(theta[agent])
NE_gaps = []
for i in tqdm(range(N)):
    NE_gap = []
    next_pi = np.copy(pi)
    next_theta = np.copy(theta)
    for agent in range(n):
        r = np.copy(reward)
        r = np.transpose(r)
        for left_agent in range(agent):
            r = np.matmul((r), pi[left_agent])
        r = np.transpose(r)
        for right_agent in range(n-1,agent, -1):
            r = np.matmul(r, pi[right_agent])
        r = r-np.dot(r,pi[agent])
        NE_gap.append(np.max(r))
        
        next_pi[agent] = pi[agent]*np.exp(eta*(r + tau/pi[agent] - tau*action_spaces[agent]) )
        next_pi[agent]/=np.sum(next_pi[agent])
    NE_gaps.append(NE_gap)
    pi = np.copy(next_pi)
    theta = np.copy(next_theta)

NE_gaps = np.array(NE_gaps)
NPG_log_barrier_NE_gaps = np.sum(NE_gaps, axis=1)
# plt.plot(NPG_log_barrier_NE_gaps)

def projection_simplex_sort(v, z=1):
	# Courtesy: EdwardRaff/projection_simplex.py
    if v.sum() == z and np.alltrue(v >= 0):
        return v
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

eta = 0.1
tau = 0
pi_star = np.copy(initial_pi)
new_pi_star = np.copy(initial_pi)

pi = np.copy(initial_pi)
theta = np.copy(initial_pi)
for agent in range(n):
    theta[agent] = np.log(theta[agent])

NE_gaps = []
N = 10000
for i in tqdm(range(N)):
    NE_gap = []
    next_pi = np.copy(pi)
    next_theta = np.copy(theta)
    for agent in range(n):
        r = np.copy(reward)
        r = np.transpose(r)
        for left_agent in range(agent):
            r = np.matmul((r), pi[left_agent])
        r = np.transpose(r)
        for right_agent in range(n-1,agent, -1):
            r = np.matmul(r, pi[right_agent])
        r = r-np.dot(r,pi[agent])
        ne = r-np.dot(r,pi[agent])
        NE_gap.append(np.max(ne))
        
        next_pi[agent] = projection_simplex_sort(pi[agent]+eta*r)
        next_pi[agent]/=np.sum(next_pi[agent])
    NE_gaps.append(NE_gap)
    pi = np.copy(next_pi)

NE_gaps = np.array(NE_gaps)
PGA_NE_gaps = np.sum(NE_gaps, axis=1)

end = 5000
clrs = sns.color_palette("husl", 6)
clrs =sns.color_palette("Dark2")
fig1 = plt.figure(figsize=(6,4))


plt.plot(PGA_NE_gaps[:end], color = clrs[0], label="Direct Projected Q Ascent")
plt.plot(PG_NE_gaps[:end], color = clrs[2],label="Softmax PG")
plt.plot(NPG_NE_gaps[:end], color = clrs[3],label="Softmax NPG")
plt.plot(NPG_log_barrier_NE_gaps[:end],color = clrs[4], label="Softmax NPG log barrier")
plt.plot(NPG_regularization_NE_gaps[:end], color = clrs[5],label="Softmax NPG Entropy Reg.")
plt.xlabel("Iterations")
plt.ylabel("NE-gap")
plt.legend()
plt.tight_layout()
plt.show()

