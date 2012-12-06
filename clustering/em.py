#!/usr/bin/env python

import math
import numpy as np
import random

MAX_ITERATIONS = 100
CONVERGENCE_THRESHOLD = 1e-4
VERY_SMALL_NUMBER = -1e12

np.seterr(all='raise')

def cluster(data, pdf, init_params, update_params, k=4):
    
    log_likelihood, posterior, parameters = expectation_maximization(data, pdf, init_params, update_params, k)
    # TODO(salspaugh): The "centers" computed below are not actually central. 
    # It's obvious if you think about it.
   
    clusters = np.apply_along_axis(np.argmax, 1, posterior) 
    centers = [np.argmax(posterior[:,c]) for c in range(k)]
    for c in range(k):
        clusters[clusters == c] = centers[c]
    #print centers
    return clusters, centers

def expectation_maximization(data, pdf, init_parameters, update_parameters, k):
    
    n = data.shape[0] # number of observations
    d = data.shape[1] # dimensionality of observations

    prior = initialize_prior(k) # probability of being in each class
                                # initially this is uniform   
    posterior = np.zeros((n,k)) # probability of being in a class given a data point
    emissions = np.zeros((n,k)) # probability of a data point given the class it's in
    
    parameters = init_parameters(k, d)
    #(mu, sigma) = parameters
    curr_ll = VERY_SMALL_NUMBER # very unlikely (log likelihood is negative)
   
    iterations = 0.
    for _ in range(MAX_ITERATIONS):
        prev_ll = curr_ll
        #print prior
        #print posterior
        #print emissions
        curr_ll = do_EM_iteration(k, data, emissions, prior, posterior, pdf, update_parameters, *parameters)
        #print curr_ll
        iterations += 1.
        if converged(prev_ll, curr_ll):
            print "Converged in ", iterations , "iterations"
            break

    #print "Means of component densities:\n", mu
    #print "Priors:\n", prior
    #print "Sigmas:\n", sigma
    
    return curr_ll, posterior, parameters 

def initialize_prior(k, uniform=False):
    if uniform:
        return np.ones(k) / k
    else: # Sometimes uniform will get you stuck? Or so I've read.
        prior = np.random.rand(k)
        return prior / prior.sum()

def do_EM_iteration(k, data, emissions, prior, posterior, pdf, update_parameters, *parameters):

    n = data.shape[0]

    # E-step
    update_emissions(k, data, emissions, pdf, *parameters) 
    update_posterior(emissions, prior, posterior)
    log_likelihood = compute_log_likelihood(posterior)
    normalize_posterior(posterior)

    # M-step
    update_prior(prior, posterior)
    update_parameters(data, prior, posterior, *parameters)

    return log_likelihood

def update_emissions(k, data, emissions, pdf, *parameters):
    n = data.shape[0]
    for data_idx in range(n):
        emissions[data_idx] = emission_probabilities(k, data[data_idx], pdf, *parameters)

def emission_probabilities(k, data, pdf, *parameters):
    per_class_probabilities = []
    for cls in range(k):
        per_class_probabilities.append(pdf(data, cls, *parameters))
    return per_class_probabilities

def update_posterior(emissions, priors, posterior):
    n = emissions.shape[0] # number of observations
    k = priors.shape[0] # number of classes
    for d in range(n):
        for s in range(k):
            posterior[d][s] = priors[s] * emissions[d][s]

def compute_log_likelihood(posterior):
    return np.log(posterior.sum(axis=1)).sum()

def normalize_posterior(posterior):
    n = posterior.shape[0]
    k = posterior.shape[1]
    for d in range(n):
        sample_sum = float(sum(posterior[d]))
        if sample_sum != 0:
            for s in range(k):
                posterior[d][s] = posterior[d][s] / sample_sum

def update_prior(prior, posterior):
    k = prior.shape[0]
    n = posterior.shape[0]
    for s in range(k):
        sample_sum = 0.
        for x in range(n):
            sample_sum += posterior[x][s]
        prior[s] = sample_sum / n

def converged(old_log_likelihood, new_log_likelihood):
    return (new_log_likelihood - old_log_likelihood < CONVERGENCE_THRESHOLD)

# Distribution-specific functions:

# Multinomial
def multinomial_parameter_init(k, d):
    p = np.random.rand(k,d,d)
    # p[x,y,k] = P_k(x|y) = prob of seeing x after y in class k
    # the columns should sum to unity for each class
    for c in range(k):
        for j in range(d): # normalize across columns
            p[c,:,j] = p[c,:,j] / p[c,:,j].sum() 
    return p,

def multinomial_pdf(counts, cls, *parameters):
    p = parameters[0][cls]
    d = p.shape[0]
    # counts[x,y] = c when x appears after y c times in observation  
    return np.power(p, counts.reshape(d,d)).prod() 

def multinomial_parameter_update(data, prior, posterior, *parameters):
    n = data.shape[0]
    k = prior.shape[0]
    d = data.shape[1] 
    p = parameters[0]

    for cls_idx in range(k):
        prob = np.zeros((d,d))
        for obs_idx in range(n):
            prob += (data[obs_idx].reshape(d,d))*(posterior[obs_idx][cls_idx])
        p[cls_idx] = prob     
        for j in range(d): # normalize across columns
            normalization = p[cls_idx,:,j].sum()
            if normalization > 0:
                p[cls_idx,:,j] = p[cls_idx,:,j] / p[cls_idx,:,j].sum()

# Isotropic Bivariate Normal
def isotropic_bivariate_normal_parameter_init(k, d):
    mu = np.zeros((k,d))
    mu[0,:] = (0.15, 0.231)
    mu[1,:] = (-0.121, 0.435)
    mu[2,:] = (-0.489, -0.890)
    mu[3,:] = (0.98, -0.678)
    sigma = np.ones((k,d))
    return mu, sigma

def isotropic_bivariate_normal_pdf(point, cls, *parameters):
    mu = parameters[0][cls]
    sigma = parameters[1][cls]
    exp1 = -1.*(point[0] - mu[0])*(point[0] - mu[0]) / (2.*sigma[0])
    exp2 = -1.*(point[1] - mu[1])*(point[1] - mu[1]) / (2.*sigma[1])
    dr1 = math.sqrt(2.*math.pi*sigma[0])
    dr2 = math.sqrt(2.*math.pi*sigma[1])
    return math.exp(exp1) * math.exp(exp2) * (1. / dr1) * (1. / dr2)

def isotropic_bivariate_normal_parameter_update(data, prior, posterior, *parameters):
    k = prior.shape[0] 
    n = data.shape[0]

    (mu, sigma) = parameters

    # Update mu = posterior(x) * obs / sum_t(posterior(x))
    for s in range(k):
        prob1 = 0.
        prob2 = 0.
        for d in range(n):
            prob1 = prob1 + (posterior[d][s] * data[d][0])
            prob2 = prob2 + (posterior[d][s] * data[d][1])
        dr = float(prior[s]*n)
        mu[s] = (prob1/dr, prob2/dr)

    # Update sigma = posterior(x) * (obs - mu) (obs - mu)' / sum_t(posterior(x))
    for s in range(k):
        nr = 0.
        for x in range(n):
            omu = (data[x][0] - mu[s][0], data[x][1] - mu[s][1])
            omu_omut = (omu[0]*omu[0]) + (omu[1]*omu[1])
            nr += float(posterior[x][s]*omu_omut)
        dr = prior[s]*2.*n 
        sigma[s] = (nr / dr, nr / dr)
