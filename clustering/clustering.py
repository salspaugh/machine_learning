#!/usr/bin/env python

import em
import kmedoids
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import spectral
from splparser.parser import parse as splparse
from splparser.parser import SPLSyntaxError
from zss.compare import distance as tree_dist

TEST_FILE = "points.dat"
TEST_POINTS = 1000

def main():
    np.set_printoptions(linewidth=200, threshold=1000000)
    
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-k", "--num_clusters", dest="k", type="int",
                      help="number of clusters to look for")
    parser.add_option("-f", "--file", dest="file", type="string", metavar="FILE",
                      help="file containing the data to cluster (supported types: splunk queries and x,y coordinates)")
    parser.add_option("-e", "--em", 
                      action="store_true", default=False, dest="em",
                      help="run expectation-maximization on a data in FILE (defaults to n data points of four Gaussian x,y clusters)")
    parser.add_option("-m", "--kmedoids", 
                      action="store_true", default=False, dest="kmedoids",
                      help="run k-medoids on a data in FILE (defaults to n data points of four Gaussian x,y clusters)")
    parser.add_option("-s", "--spectral", 
                      action="store_true", default=False, dest="spectral",
                      help="run spectral clustering on data in FILE (defaults to n data points of four Gaussian x,y clusters)")
    parser.add_option("-q", "--queries", 
                      action="store_false", default=True, dest="euclidean",
                      help="assume data is SPL queries (defaults to Euclidean x,y points)")
    (options, args) = parser.parse_args()

    precomputed_distances = False
    f = (TEST_FILE if options.file is None else options.file)
    if is_npy(f):
        r = read_distance_matrix
        precomputed_distances = True
    else:
        r = (read_points if options.euclidean else read_queries)
    k = (4 if options.k is None else options.k)
    
    
    clusterer = kmedoids.cluster
    
    if options.em and not options.kmedoids and not options.spectral:
        clusterer = em.cluster
        pdf = em.isotropic_bivariate_normal_pdf if options.euclidean else None
        param_init = em.isotropic_bivariate_normal_parameter_init \
                                if options.euclidean else None
        param_update = em.isotropic_bivariate_normal_parameter_update \
                                if options.euclidean else None
    
    else:
        d = build_euclidean_distance_matrix if options.euclidean \
                                            else build_tree_edit_distance_matrix
        if options.kmedoids and not options.spectral and not options.em:
            clusterer = kmedoids.cluster
        elif options.spectral and not options.kmedoids and not options.em:
            clusterer = spectral.cluster
        else:
            print "Please pick one clustering method."
            exit()
    
    distances = data = get_data(datafile=f, datareader=r)
    if not options.em:
        if not precomputed_distances: # TODO(salspaugh): Take distances file as an argument.
            distances = compute_distances(data, distancer=d, savefile='distances.npy') 
        clusters, centers = clusterer(distances, k=k)
    else:
        clusters, centers = clusterer(data, pdf, param_init, param_update, k=k)
    
    output_results(data, clusters, centers, plot=options.euclidean)
   
    # TODO(salspaugh): Clean up option parsing!
    # TODO(salspaugh): Data input error handling.

def is_npy(filename):
    return (filename[filename.rfind('.'):] == '.npy')

def read_distance_matrix(datafile):
    return np.mat(np.load(datafile))

def read_points(datafile):
    points = []
    with open(datafile) as data:
        for line in data.readlines():
            (x,y) = line.split(',')
            point = (float(x), float(y))
            points.append(point)
    return np.array(points)

def generate_random_point_clusters(datafile, n=TEST_POINTS): 
    points = []
    with open(datafile, 'w') as out:
        sign = [-1, 1]
        mu = 4
        sigma = 1
        for _ in range(n):
            xsign = random.choice(sign)
            ysign = random.choice(sign)
            x = random.gauss(xsign*mu, sigma)
            y = random.gauss(ysign*mu, sigma)
            s = "%3.2f, %3.2f\n" % (x, y)
            points.append((x,y))
            out.write(s)
            out.flush()
    return points

def read_queries(datafile):
    parsetrees = []
    with open(datafile) as data:
        for query in data.readlines():
            try:
                query = query.strip('\n')
                parsetree = splparse(query)
                parsetrees.append(parsetree)
            except SPLSyntaxError:
                print "Syntax error encountered while parsing SPL."
                print "\t" + query
                continue
    print "Done parsing queries."
    return parsetrees

def build_tree_edit_distance_matrix(parsetrees):
    return build_distance_matrix(parsetrees, tree_dist)

def build_euclidean_distance_matrix(points):
    return build_distance_matrix(points, euclidean_dist)

def euclidean_dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def build_distance_matrix(data, distfn):
    m = len(data)
    distances = np.zeros((m,m))
    i = j = 0.
    for i in range(m):
        for j in range(i+1, m): # The distance matrix is symmetric.
            p = data[i]
            q = data[j]
            distances[i,j] = distances[j,i] = distfn(p, q)
    return distances

def get_data(datafile=TEST_FILE, datareader=read_points):
    try:
        data = datareader(datafile)
    except IOError:
        data = generate_random_point_clusters(datafile)
    return data

def compute_distances(data, distancer=build_euclidean_distance_matrix, savefile=None):
    distances = distancer(data) 
    if not savefile is None:
        np.save(savefile, distances)    
    return distances

def output_results(data, clusters, centers, plot=False):
    pass
    print_results(data, clusters) # TODO: Check if this works for queries.
    if plot:
        plot_results(data, clusters, centers)

def print_results(data, clusters):
    for i in range(len(data)):
        print clusters[i], data[i]

def plot_results(data, cluster_idxs, medoid_idxs):
    data = np.array(data)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    cluster_idx_count = 0
    for medoid_idx in medoid_idxs:
        cluster_idx = np.where(cluster_idxs == medoid_idx)[0]
        print cluster_idx
        x = [pt[0] for pt in data[cluster_idx]]
        y = [pt[1] for pt in data[cluster_idx]]
        plt.plot(x, y, color=colors[cluster_idx_count], marker='o', linestyle='None')
        plt.plot(data[medoid_idx][0], data[medoid_idx][1], color='k', marker='o', linestyle='None')
        cluster_idx_count += 1
    plt.savefig('clusters.png')

if __name__ == "__main__":
    main()
