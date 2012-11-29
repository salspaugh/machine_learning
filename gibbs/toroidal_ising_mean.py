#!/usr/bin/env python

import math
import numpy as np
import random
import sys

# Core functionality:

def ps7q1():

    # Problem parameters:
    sqrtd = 7
    d = sqrtd*sqrtd         # number of nodes
    
    burn_in_period = 1000
    n = 5000                # number of samples
    theta_edges = (lambda x: .25)
    #zero_one_init = (lambda x: max((-1.)**x, 0.)) # for testing
    neg_pos_one_init = (lambda x: (-1.)**x)
    #theta_node = zero_one_init
    theta_node = neg_pos_one_init
    #x = (lambda x: random.uniform(0, 1.))
    x = (lambda x: random.uniform(-1., 1.))

    # Do Gibb's sampling.
    gibbs_grid = create_toroidal_grid(sqrtd, x, theta_node, theta_edges)
    mu = moments_via_gibbs(gibbs_grid, burn_in_period, n, d)
    
    print "\nGibb's samples moments:"
    for (id, node) in gibbs_grid.iteritems():
        print "ID: %d, mean = %3.4f" % (id, mu[0,id-1])

    # Do naive mean field updates.
    nmf_grid = create_toroidal_grid(sqrtd, x, theta_node, theta_edges)
    tau = moments_via_naive_mean_field(nmf_grid, 2*n)
    
    print "\nNaive mean field moments:"
    #for (id, node) in nmf_grid.iteritems():
    #    print "ID: %d, mean = %3.4f" % (id, tau[id-1])
    id = 1
    for i in range(sqrtd):
        print ""
        for j in range(sqrtd):
            sys.stdout.write("%3.4f\t" % tau[id-1])
            id += 1
    print ""

    # Error:
    print "\nError:"
    print np.absolute(tau - mu).sum() / float(d)

def create_toroidal_grid(sqrtd, value, theta_node, theta_edge):
    n = sqrtd * sqrtd
    grid = {}
    for i in range(1, n+1):
        grid[i] = IsingNode(i, value(i), theta_node(i), theta_edge(i))
    for (id, node) in grid.iteritems():
        node.neighbors = get_toroidal_neighbors(id, sqrtd, grid)
    return grid

def moments_via_gibbs(grid, burn_in_period, n, d):
    # Burn-in period.
    for t in range(burn_in_period):
        do_gibbs_burn_in_iteration(grid)
    
    # Take samples.
    samples = np.matrix([[0.]*d]*n) # (n x d)
    for t in range(n):
        samples[t,:] = do_gibbs_sample_iteration(grid)
    
    return np.array(samples.sum(axis=0) / float(n))

def do_gibbs_burn_in_iteration(grid):
    for (id, node) in grid.iteritems():
        node.do_gibbs_update()

def do_gibbs_sample_iteration(grid):
    sample = np.array([0.]*len(grid.values()))
    for (id, node) in grid.iteritems():
        node.do_gibbs_update()
        sample[id-1] = node.value
    return sample

def moments_via_naive_mean_field(grid, n):
    # Update the mean for a while.
    for t in range(n):
        do_nmf_iteration(grid)
    
    return np.array([node.mean for node in grid.values()])
    
def do_nmf_iteration(grid):
    for (id, node) in grid.iteritems():
        node.do_nmf_update()

class IsingNode(object):
    
    def __init__(self, id, value, theta_node, theta_edge):
        self.id = int(id)
        self.value = value
        self.mean = value
        self.theta_node = theta_node
        self.theta_edge = theta_edge
        self.neighbors = {'up' : None,
                          'down' : None,
                          'left' : None,
                          'right' : None}

    def __cmp__(self, other):
        return self.id == other.id

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return "ID: %d, neighbors = {%s, %s, %s, %s}" % (self.id, 
                                                        str(self.neighbors['up']), 
                                                        str(self.neighbors['down']), 
                                                        str(self.neighbors['left']), 
                                                        str(self.neighbors['right'])) 

    def zero_one_rule(self):
        neighbor_sum = sum([n.value for n in self.neighbors.values()])
        return (1. / (1. + math.exp(-1.*(self.theta_node + self.theta_edge*neighbor_sum))))
    
    def negative_positive_one_rule(self):
        neighbor_sum = sum([n.value for n in self.neighbors.values()])
        #return (1. / (1. + math.exp(-2.*(self.theta_edge*neighbor_sum - 2.*self.theta_node)))
        return (1. / (1. + math.exp(-2.*(self.theta_node + self.theta_edge*neighbor_sum))))

    def do_gibbs_update(self): # FIXME
        u = random.uniform(0.,1.)
        thresh = self.negative_positive_one_rule()
        #thresh = self.zero_one_rule()
        self.value = (1. if (u <= thresh) else -1.)
    
    def do_nmf_update(self):
        neighbor_sum = sum([n.mean for n in self.neighbors.values()])
        y = math.exp(2.*(self.theta_node + self.theta_edge*neighbor_sum)) 
        self.mean = ((y - 1.)/(1. + y))

# Necessary but not particularly interesting functions:

def get_toroidal_neighbors(id, sqrtd, grid):
    right_edge = is_right_edge(id, sqrtd)
    left_edge = is_left_edge(id, sqrtd)
    top_edge = is_top_edge(id, sqrtd)
    bottom_edge = is_bottom_edge(id, sqrtd)
    row = get_row(id, sqrtd)
    col = get_col(id, sqrtd)
    up = ((id - sqrtd) if not top_edge else ((sqrtd * sqrtd) - sqrtd + col))
    down = ((id + sqrtd) if not bottom_edge else col)
    left = ((id - 1) if not left_edge else ((id - 1) + sqrtd))
    right = ((id + 1) if not right_edge else ((id + 1) - sqrtd))    
    return {'up' : grid[up], 
            'down' : grid[down],
            'left' : grid[left],
            'right' : grid[right]}

def is_right_edge(id, sqrtd):
    return int((id + 1) % sqrtd == 1)

def is_left_edge(id, sqrtd):
    return int((id - 1) % sqrtd == 0)

def is_top_edge(id, sqrtd):
    return int((id - sqrtd) <= 0)

def is_bottom_edge(id, sqrtd):
    return int((id + sqrtd) > (sqrtd * sqrtd))

def get_row(id, sqrtd):
    return ((int(id) - 1) / int(sqrtd)) + 1

def get_col(id, sqrtd):
    return (int(id) % int(sqrtd)) + (sqrtd * is_right_edge(id, sqrtd)) 
 
if __name__ == "__main__":
    ps7q1()
