import copy
import numpy as np
import random as rand
import math

def HBA(SearchAgents_no, Max_iter, fobj, dim, lb, ub):
    '''
    This code is for Honey Badger Algorithm(HBA) to solve minimization problem.
    "SearchAgents_no" is number of searching agents
    "Max_iter" is limit of iteration
    "fobj" denotes the objective function
    "dim" is number of split num
    "lb" is lower bound
    "ub" is upper bound
    '''
    # initialize position vector and score for the prey
    Prey_pos = np.zeros(dim)

    # change this to -inf for maximization problems
    Prey_score = float('inf')
    Prey_score_pre = Prey_score

    # tolerance to stop the algorithm
    delta = 1e-6
    Flag = 0

    # Initialize the positions of search agents. Size is SearchAgents_no x dim
    Positions = np.zeros((SearchAgents_no, dim))
    Fitnesses = np.zeros(SearchAgents_no)
    for i in range(SearchAgents_no):
        for j in range(dim):
            Positions[i][j] = lb + rand.random() * (ub - lb)
        # Calculate objective function for each search agent
        Fitnesses[i] = fobj(Positions[i])
        if Fitnesses[i] < Prey_score: # Change this to > for maximization problem
            Prey_score = Fitnesses[i]
            Prey_pos = copy.deepcopy(Positions[i])
            
    # Loop counter
    iter = 0

    # Constant values
    C = 2.0
    beta = 6.0
    epsilon = 2e+25

    ''' Main loop '''
    while iter < Max_iter and Flag <= 3:
        # Update density factor
        alpha = C * math.exp((-iter)/Max_iter)
                
        # Search the next positions
        for i in range(SearchAgents_no):
            r = rand.random()
            F = 1 if rand.random() <= 0.5 else -1
            x_new = np.zeros(dim)

            # Defining intensity
            S = math.dist(Positions[i], Positions[i+1]) if i != len(Positions)-1 else math.dist(Positions[i], Positions[0])
            S = S**2
            d_i = math.dist(Positions[i], Prey_pos)
            d_i = epsilon if d_i==0 else d_i
            I = rand.random() * S / (4*math.pi*(d_i**2))
            for j in range(dim):
                # Digging phase
                if r < 0.5:
                    r3 = rand.random()
                    r4 = rand.random()
                    r5 = rand.random()
                    x_new[j] = Prey_pos[j] + F * beta  * I * Prey_pos[j] + F * r3 * alpha * d_i * abs(math.cos(2*math.pi*r4)*(1-math.cos(2*math.pi*r5))) 
                # Honey phase
                else:
                    r7 = rand.random()
                    x_new[j] = Prey_pos[j] + F * r7 *  alpha * d_i

            # Return back the search agents that go beyond the boundaries of the search space
            x_new = np.clip(x_new, lb, ub)

            # Update Prey_score, Prey_pos
            f_new = fobj(x_new)
            if f_new < Fitnesses[i]:
                Fitnesses[i] = f_new
                Positions[i] = copy.deepcopy(x_new)
                if f_new < Prey_score:
                    Prey_score = f_new
                    Prey_pos = copy.deepcopy(x_new)

        # increase the iteration index by 1
        iter = iter + 1

        # check to see whether the stopping criterion is satisifed
        if abs(Prey_score - Prey_score_pre) < delta:
            Flag = Flag + 1
        else:
            Prey_score_pre = Prey_score
    
    return Prey_pos