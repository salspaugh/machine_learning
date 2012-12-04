#!/usr/bin/env python

import math
import numpy as np
import random

MAX_ITERATIONS = 10000
CONVERGENCE_THRESH = 1e-14

def cluster(data, likelihood_fn, initial_parameters, k=4):
    posterior = expectation_maximization(data, likelihood_fn, initial_parameters, k=k)
    #print posterior
    clusters = np.apply_along_axis(np.argmax, 1, posterior) 
    return clusters, [0.]*k

def expectation_maximization(data, likelihood_fn, initial_parameters, k=4):
   
    n = data.shape[0] # number of observations
    d = data.shape[1] # dimensionality

    prior = initialize_cluster_probabilities(k)
    parameters = initial_parameters(data, k)
    curr_ll = compute_log_likelihood(likelihood_fn, data, prior, *parameters)
    
    for _ in range(MAX_ITERATIONS):
        
        # E step
        likelihood = likelihood_fn(data, prior, *parameters)
        posterior = compute_posterior(likelihood)
        
        # M step
        prior = update_cluster_probabilities(posterior)
        parameters = update_parameters(data, prior, posterior)
        print parameters
        print "prior", prior
        print "posterior0", posterior[0,:]
        print "posterior0", posterior[0,:]
        #print 'in main', parameters
        next_ll = compute_log_likelihood(likelihood_fn, data, prior, *parameters)
        print curr_ll, next_ll
        
        if converged(curr_ll, next_ll):
            break
        
        curr_ll = next_ll
    return posterior

def initialize_cluster_probabilities(k):
    return np.ones((k,1)) / float(k) # uniform

def compute_log_likelihood(likelihood_fn, data, prior, *parameters):
    #print 'in compute_log_likelihood', parameters
    likelihood = likelihood_fn(data, prior, *parameters)
    return np.log(likelihood.sum(axis=1)).sum()

def compute_posterior(likelihood):
    return likelihood / np.transpose(np.matrix(likelihood.sum(axis=1)))

def update_cluster_probabilities(posterior):
    n = posterior.shape[0]
    return np.transpose(posterior.sum(axis=0)) / n

def update_parameters(data, prior, posterior):
    #print "UPDATE PARAMETERS"
    n = data.shape[0]
    d = data.shape[1]
    k = posterior.shape[1]
    mu = np.ones((k,d))
    sigmasq = np.ones((k,1))
    for c in range(k):
        dp = np.ones((n,d))
        for i in range(n):
            dp[i,:] = data[i,:]*posterior[i,c]
        mu[c,:] = dp.sum(axis=0) / (n*prior[c,:])
        diff = data - mu[c,:]
        sigmasq[c,:] = np.multiply(diff, diff).sum(axis=1).sum() / n  
    return mu, sigmasq

def converged(old, new):
    return (new - old <= CONVERGENCE_THRESH)

# For various distributions ...
def isotropic_bi_normal_likelihood(data, prior, *parameters):
    mu = parameters[0]
    sigmasq = parameters[1]
    
    n = data.shape[0]
    k = mu.shape[0]
    d = data.shape[1]

    class_likelihood = np.zeros((n,k))
    z = 2.*math.pi*sigmasq

    for c in range(k):
        distances = np.matrix(data - mu[c,:])
        for j in range(n):
            distancessq = distances[j,:]*np.transpose(distances[j,:])
            exponent = np.exp(-1.*distancessq/(2.*sigmasq[c]))
            class_likelihood[j,c] = prior[c]*(1./z[c])*exponent

    return class_likelihood

def isotropic_bi_normal_initial_parameters(data, k):
    n = data.shape[0]
    d = data.shape[1]

    mu = data[[random.randint(0, n - 1) for _ in range(k)], :] # (k x d)

    diff = data - mu[0,:]
    s = np.multiply(diff, diff).sum(axis=1).sum() / n  
    sigmasq = np.ones((k,1))*s # isotropriorc => cov = sigma*I

    mu = np.matrix([[.15, .231],
                   [-.121, .435],
                   [-.489, -.890],
                   [.98, -.678]])
    sigmasq = np.ones((k,1))

    return mu, sigmasq
