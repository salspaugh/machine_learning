#!/usr/bin/env python

import math
import numpy as np
import random

MAX_ITERATIONS = 100
CONVERGENCE_THRESHOLD = 1e-4

def cluster(data, likelihood_fn, new_parameters, k=4):
    n = data.shape[0] # number of observations
    posterior, parameters = expectation_maximization(data, likelihood_fn, new_parameters, k)
    # TODO(salspaugh): some stuff to make the clusters
    clusters = [0.]*n
    centers = [0.]*k
    return clusters, centers

def expectation_maximization(data, likelihood_fn, new_parameters, k):
    n = data.shape[0] # number of observations
    d = data.shape[1] # dimensionality of observations

    #prior = map(lambda x: x/k, [1.]*k)
    prior = np.ones(k) / k

    #mu = [(0.,0.)]*k
    mu = np.zeros((k,d))

    #mu[0] = (0.15, 0.231)
    #mu[1] = (-0.121, 0.435)
    #mu[2] = (-0.489, -0.890)
    #mu[3] = (0.98, -0.678)
    mu[0,:] = (0.15, 0.231)
    mu[1,:] = (-0.121, 0.435)
    mu[2,:] = (-0.489, -0.890)
    mu[3,:] = (0.98, -0.678)

    #sigma = [(1., 1.)]*k 
    sigma = np.ones((k,d))

    parameters = (mu, sigma)
    posterior = None
    curr_ll = 0.
    
    for _ in range(MAX_ITERATIONS):
        prev_log_lik = curr_ll
        curr_ll = emstep(data, prior, mu, sigma, k)
        if (abs(curr_ll - prev_log_lik) < CONVERGENCE_THRESHOLD):
            break

    print "Means of component densities:\n", mu
    print "Priors:\n", prior
    print "Sigmas:\n", sigma
     
    return posterior, parameters 

def isotropic_bivariate_normal_pdf(x, mu, sigma):
    exp1 = -1.*(x[0] - mu[0])*(x[0] - mu[0]) / (2.*sigma[0])
    exp2 = -1.*(x[1] - mu[1])*(x[1] - mu[1]) / (2.*sigma[1])
    dr1 = math.sqrt(2.*math.pi*sigma[0])
    dr2 = math.sqrt(2.*math.pi*sigma[1])
    return math.exp(exp1) * math.exp(exp2) * (1. / dr1) * (1. / dr2)

def isotropic_bivariate_normal_parameter_update():
    pass

def get_obs_likelihood(obs, mu, sigma):
    k = mu.shape[0]
    pdfs = []
    for i in range(k):
        pdfs.append(isotropic_bivariate_normal_pdf(obs, mu[i], sigma[i]))
    return pdfs

def normalize(vec):
    sum = float(sum(vec))
    if (sum == 0.):
        sum = 1.0
    return map(lambda x: x/sum, vec)

def emstep(data, prior, mu, sigma, k):

    num_samples = len(data)
    loglik = 0.
  
    # E-step
    # First compute the activations for each sample point
    activations = []
    for x in range(num_samples):
        a = []
        for s in range(k):
            a.append((0.,0.))
        activations.append(a)
    for x in range(num_samples):
        activations[x] = get_obs_likelihood(data[x], mu, sigma)

    # Compute the log-likelihood by multiplying activations with the prior
    for d in range(num_samples):
        prob_s = 0.
        for s in range(k):
            prob_s = prob_s + (activations[d][s] * prior[s])
        loglik = loglik + math.log(prob_s)

    # Now calculate the posterior distribution for each sample point
    post = []
    for x in range(num_samples):
        post.append([0.]*k)
    for d in range(num_samples):
        for s in range(k):
            post[d][s] = prior[s] * activations[d][s]
    
    # Normalize posterior for each sample
    # This will make the posterior for each sample to add up to 1.
    for d in range(num_samples):
        sample_sum = float(sum(post[d]))
        if sample_sum != 0:
            for s in range(k):
                post[d][s] = post[d][s] / sample_sum

    #print "Posterior:\n", post[0]
    # M-step
    # Update the prior by using the posterior
    for s in range(k):
        sample_sum = 0.
        for x in range(num_samples):
            sample_sum += post[x][s]
        prior[s] = sample_sum / num_samples
        
    # Update mu = post(x) * obs / sum_t(post(x))
    for s in range(k):
        prob1 = 0.
        prob2 = 0.
        for d in range(num_samples):
            prob1 = prob1 + (post[d][s] * data[d][0])
            prob2 = prob2 + (post[d][s] * data[d][1])
        dr = float(prior[s]*num_samples)
        mu[s] = (prob1/dr, prob2/dr)

    # Update sigma = post(x) * (obs - mu) (obs - mu)' / sum_t(post(x))
    for s in range(k):
        nr = 0.
        for x in range(num_samples):
            omu = (data[x][0] - mu[s][0], data[x][1] - mu[s][1])
            omu_omut = (omu[0]*omu[0]) + (omu[1]*omu[1])
            nr += float(post[x][s]*omu_omut)
        dr = prior[s]*2.*num_samples 
        sigma[s] = (nr / dr, nr / dr)

    return loglik

