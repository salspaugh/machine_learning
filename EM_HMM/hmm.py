#!/usr/bin/env python
# Description: A program to find the MLEs of an HMM with bivariate emission
#   probabilities and m states using Expectation Maximization. In fact 
#   implements the Baum-Welch algorithm as described in Rabiner's tutorial
#   with normalization of the forward and backward variables. Another way
#   to normalize is to use logarithms to handle underflow issues [1,2].
# References:
# [1] Rabiner, Lawrence. "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition". 1989.
# [2] Mann, Tobias. "Numerically Stable Hidden Markov Model". 2006.
# Author: Sara Alspaugh
# Date: 30 October 2012


from collections import defaultdict
import numpy as np

LOGZERO = 'LOGZERO'

def eexp(x):
    if x == LOGZERO:
        return 0.
    else:
        return np.exp(x)

def eln(x):
    if x == 0:
        return LOGZERO
    else:
        return np.log(x)

def elnsum(x, y):
    if x == LOGZERO or y == LOGZERO:
        if x == LOGZERO:
            return y
        else:
            return x
    elif x > y:
        return x + eln(1 + np.exp(y - x))
    else:
        return y + eln(1 + np.exp(x - y))

def elnproduct(x, y):
    if x == LOGZERO or y == LOGZERO:
        return LOGZERO
    else:
        return x + y

class HMM(object):

    def __init__(self, m, training_data, test_data):
        """
        m       the number of states
        A       an mxm (from x to) transition matrix 
                A[i,j] is the probability of transitioning from state i to state j    
        pi      an m-length vector of initial state probabilities
                pi[i,:] is the probability of starting in state i 
        b       an m-length vector of emission probabilities
                updated every timestep
                assume emission probabilies are bivariate normal
                represents the probability of an observation given the current state
        mu      an 2xm matrix of bivariate normal means for the emission probabilities
        sigma   an m-length array of 2x2 bivariate normal covariances for the emission 
                probabilities
                valid because we're assuming isotropic covariance matrices
                for simplicity
        alpha   an m-length vector of forward variables updated every timestep
                ...
        beta    an m-length vector of backward variables updated every timestep
                ...
        gamma   an m-length vector updated every timestep
                the probability of being in state i given the observations
        delta   an m-length vector updated every timestep
                the highest probability along a path ending in state i given
                the observations
        psi     an m-length vector updated every timestep
                the delta for the state j preceeding state i's current delta
                in other words, the argument which maximizes delta_t(i)*a_ij
        xi      an mxm matrix updated every timestep which represents the 
                probability of being in state i now and state j in the next
                timestep
        """

        self.m = int(m)                                                 # m
        
        self.transition_probs = np.matrix([[1./m]*m]*m)                 # A
        self.initial_probs = np.matrix([[1./m]]*m)                      # pi

        self.emission_probs = defaultdict(self.column_vector_matrix) # HACK  
        self.emission_probs[0] = np.matrix([[1./m]]*m)                  # b
        #self.mean = np.matrix([[0.]*m]*2)                              # bad mu --- doesn't work
        self.mean = np.matrix([[2.,-2.,-2.,2.],[2.,2., -2.,-2.]])       # good mu
        self.covariance = [np.matrix([[1., 0.],[0., 1.]])]*m            # sigma

        # it doesn't matter what the following are set to because we initialize
        # them based on the above probabilities to start
        self.forwards = defaultdict(self.column_vector_matrix)          # HACK
        self.forwards[0] = np.matrix([[1.]]*m)                          # alpha        
        self.backwards = defaultdict(self.column_vector_matrix)         # HACK
        self.backwards[0] = np.matrix([[1.]]*m)                         # beta

        self.gamma = {} 
        self.delta = {} # not used TODO
        self.psi = {}   # not used TODO
        self.xi = defaultdict(self.symmetric_matrix)

        self.training_data = training_data
        self.test_data = test_data
        self.timesteps = len(training_data.keys())
  
        self.debug = False

    def column_vector_matrix(self):
        return np.matrix([[1./self.m]]*self.m)

    def symmetric_matrix(self):
        return np.matrix([[1.]*self.m]*self.m)

    def print_vector_probabilities(self, p, name, timesteps=10, end=False):
        if not end:
            for t in range(timesteps):
                print "time:", t, name, "=", p[t].transpose()
        else:
            for t in range(self.timesteps - timesteps,  self.timesteps):
                print "time:", t, name, "=", p[t].transpose()

    def print_matrix_probabilities(self, p, name, timesteps=10, end=False):
        if not end:
            for t in range(self.timesteps):
                for row in range(p.shape[0]):
                    print "time:", t, name, "=", p[t][row,:]
        else:
            for t in range(self.timesteps - timesteps, self.timesteps):
                for row in range(p[t].shape[0]):
                    print "time:", t, name, "=", p[t][row,:]

    def print_to_check(self):
        self.print_vector_probabilities(self.emission_probs, "b") 
        self.print_vector_probabilities(self.forwards, "alpha") 
        self.print_vector_probabilities(self.backwards, "beta", end=True) 
        self.print_vector_probabilities(self.gamma, "gamma", end=True)
        self.print_matrix_probabilities(self.xi, "xi", end=True)

    def bivariate_normal_pdf(self, data, mean, cov):
        k = 2 # because it's bivariate, would have higher k for multivariate
        two_pi_to_neg_half_k = np.exp(-0.5 * k * np.log(2. * np.pi))
        sqrt_det = np.power(np.linalg.det(cov), -0.5)
        dev = data - mean
        e_to_the_stuff = np.exp(-0.5 * np.dot(np.dot(dev.transpose(), np.linalg.inv(cov)), dev))
        return (two_pi_to_neg_half_k * sqrt_det * e_to_the_stuff)

    def update_emission_probabilities(self):
        for (timestep, observation) in self.training_data.iteritems():
            for state in range(self.m): # TODO: use vector operations here instead of iterating
                mu = self.mean[:,state]
                sigma = self.covariance[state]
                b = self.bivariate_normal_pdf(observation, mu, sigma)
                self.emission_probs[timestep][state,:] = b

    def update_forward_probabilities(self, log_scaling=False, normalize=True):
        if log_scaling:
            for state in range(self.m):
                self.forwards[0][state,:] = elnproduct(eln(self.initial_probs[state,:]), eln(self.emission_probs[0][state,:]))
            for t in range(self.timesteps)[1:]:
                for statej in range(self.m):
                    logalpha = LOGZERO
                    for statei in range(self.m):
                        forward_to_trans_from = elnproduct(self.forwards[t-1][statei,:], eln(self.transition_probs[statei,statej]))
                        logalpha = elnsum(logalpha, forward_to_trans_from)
                    self.forwards[t][statej,:] = elnproduct(logalpha, eln(self.emission_probs[t][statej,:]))
        else:
            self.forwards_sum = {}
            self.forwards[0] = np.multiply(self.initial_probs, self.emission_probs[0])
            self.forwards_sum[0] = np.sum(self.forwards[0])
            if normalize:
                self.forwards[0] = self.forwards[0] / self.forwards_sum[0]
            for t in range(self.timesteps)[1:]:
                sum_forwards_to_this_state = 0.
                for state in range(self.m): # TODO: change the following to use an apply for row of forwards
                    forward_to_trans_from = np.multiply(self.forwards[t-1], self.transition_probs[:,state])
                    sum_forwards_to_this_state = np.sum(forward_to_trans_from)
                    self.forwards[t][state,:] = sum_forwards_to_this_state * self.emission_probs[t][state,:]
                self.forwards_sum[t] = np.sum(self.forwards[t])
                if normalize:
                    self.forwards[t] = self.forwards[t] / self.forwards_sum[t]

    def update_backward_probabilities(self, log_scaling=False, normalize=True):
        if log_scaling:
            self.backwards[self.timesteps-1][:,:] = 0.
            for t in reversed(range(self.timesteps-1)):
                for statei in range(self.m):
                    logbeta = LOGZERO
                    for statej in range(self.m):
                        logbeta = elnsum(logbeta, elnproduct(eln(self.emission_probs[t+1]), elnproduct(self.emission_probs[t+1][statej,:], self.backwards[t+1][statej,:])))
                    self.backwards[t][statei,:] = logbeta
        else:
            self.backwards[self.timesteps-1][:,:] = 1.
            if normalize:
                self.backwards[self.timesteps-1] = self.backwards[self.timesteps-1]/np.sum(self.backwards[self.timesteps-1])
            for t in reversed(range(self.timesteps-1)):
                for state in range(self.m):
                    self.backwards[t][state,:] = np.dot(self.transition_probs[state,:], np.multiply(self.emission_probs[t+1], self.backwards[t+1])) 
                if normalize:
                    self.backwards[t] = self.backwards[t] / np.sum(self.backwards[t])

    def update_gamma(self):
       for t in range(self.timesteps):
           self.gamma[t] = np.multiply(self.forwards[t], self.backwards[t])
           self.gamma[t] = self.gamma[t] / np.sum(self.gamma[t])

    def update_delta(self): # TODO
        pass

    def update_psi(self): # TODO
        pass
    
    def update_xi(self):
        for t in range(self.timesteps-1):
            normalization = 0.
            for statei in range(self.m):
                for statej in range(self.m):
                    forward_backward = self.forwards[t][statei,:] * \
                        self.transition_probs[statei,statej] * \
                        self.emission_probs[t+1][statej,:] * \
                            self.backwards[t+1][statej,:]
                    normalization += forward_backward
                    self.xi[t][statei,statej] = forward_backward
            self.xi[t] = self.xi[t] / normalization

    def update_means(self):
        top = np.matrix([[0.]*self.m]*2)
        bottom = np.matrix([[0.]]*self.m)
        for t in range(self.timesteps):
            for state in range(self.m):
                top[:,state] += float(self.gamma[t][state,:]) * self.training_data[t]
                bottom[state,:] += self.gamma[t][state,:]
        for state in range(self.m):
            self.mean[:,state] = top[:,state] / bottom[state,:]

    def update_covariances(self):
        top = [np.matrix([[0.]*2]*2)]*self.m
        bottom = np.matrix([[0.]]*self.m)
        for t in range(self.timesteps):
            for state in range(self.m):
                distance_from_mean_squared = (self.training_data[t] - self.mean[:,state]) * (self.training_data[t] - self.mean[:,state]).transpose()
                top[state] += float(self.gamma[t][state,:]) * distance_from_mean_squared
                bottom[state,:] += self.gamma[t][state,:]
        for state in range(self.m):
            self.covariance[state] = top[state] / bottom[state,:]
    
    def update_initial_probs(self):
        for state in range(self.m):
            self.initial_probs[state,:] = self.gamma[0][state,:]

    def update_transition_probs(self):
        self.xi_sum = np.matrix([[0.]*self.m]*self.m)
        for statei in range(self.m):
            for statej in range(self.m):
                for t in range(self.timesteps-1):
                    self.xi_sum[statei, statej] += self.xi[t][statei, statej]
                self.transition_probs[statei, statej] = self.xi_sum[statei, statej]

    def update_parameters(self):
        self.update_means()
        self.update_covariances()

    def viterbi(self): # TODO: This is to handle a case I'm not sure we have
        self.update_delta()
        self.update_psi()
        self.update_gamma()

    def forward_backward(self):
        self.update_forward_probabilities()
        self.update_backward_probabilities()
        self.viterbi()
        self.update_xi()

    def E_step(self):
        self.update_emission_probabilities()
        self.forward_backward()

    def M_step(self):
        self.update_parameters()
        self.update_transition_probs()
        self.update_initial_probs()

    def compute_log_likelihood(self, data):
        self.log_likelihood = 0.
        for t in range(self.timesteps):
            self.log_likelihood += np.log(self.forwards_sum[t])

    def print_log_likelihood(self, data):
        print self.log_likelihood

    def print_means(self):
        for col in range(self.mean.shape[1]):
            print "mu" + str(col), self.mean[:,col].transpose()

    def do_baum_welch(self, iterations=10): # also known as EM for HMMs
        for iteration in range(iterations):
            self.E_step()
            self.M_step()
            if self.debug:
                self.print_to_check()
        self.compute_log_likelihood(self.training_data)
        self.compute_log_likelihood(self.test_data)
        print "Log likelihood (training data):"
        self.print_log_likelihood(self.training_data)
        print "Means (training data):"
        self.print_means()
        #print "Log likelihood (test data):"
        #self.print_log_likelihood(self.test_data)
        #print "Means (test data):

def read_ps3q1_data(filename):
    data = {}
    t = 0
    with open(filename) as datafile:
        for line in datafile.readlines():
            pts = line.split()
            pts = [float(pt) for pt in pts]
            data[t] = np.matrix(pts).transpose()
            t += 1
    return data
    
def do_ps3q1():
    training_data = read_ps3q1_data("hmm-gauss.dat")
    test_data = read_ps3q1_data("hmm-test.dat")
    hmm = HMM(4, training_data, test_data)
    hmm.do_baum_welch()

if __name__ == '__main__':
    do_ps3q1()
