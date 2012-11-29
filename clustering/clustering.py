#!/usr/bin/env python

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
    
    d = build_euclidean_distance_matrix if options.euclidean \
                                        else build_tree_edit_distance_matrix
    
    clusterer = kmedoids.cluster
    if options.kmedoids and not options.spectral:
        clusterer = kmedoids.cluster
    elif options.spectral and not options.kmedoids:
        clusterer = spectral.cluster
    else:
        print "Please pick one clustering method."
        exit()
    
    # TODO(salspaugh): Data input error handling.
    
    distances = data = get_data(datafile=f, datareader=r)
    if not precomputed_distances: # TODO(salspaugh): Take distances file as an argument.
        distances = compute_distances(data, distancer=d, savefile='distances.npy') 
    cluster(data, distances, clusterer, k=k, plot=options.euclidean)

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
    return points

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
    distances = np.matrix([[0.]*m]*m)
    i = j = 0.
    for i in range(m):
        for j in range(i+1, m): # The distance matrix is symmetric.
            p = data[i]
            q = data[j]
            distances[i,j] = distances[j,i] = distfn(p, q)
    print "Done computing queries."
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

def cluster(data, distances, clusterer, k=4, plot=False):
    clusters, centers = clusterer(distances, k=k)
    print_results(data, clusters) # TODO: Check if this works for queries.
    if plot:
        plot_results(data, clusters, centers)

def print_results(data, clusters):
    for i in range(len(data)):
        print clusters[i], data[i]

def plot_results(data, clusters, medoids):
    data = np.array(data)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    cluster_count = 0
    for medoid in medoids:
        cluster = np.where(clusters == medoid)[0]
        print cluster
        x = [pt[0] for pt in data[cluster]]
        y = [pt[1] for pt in data[cluster]]
        plt.plot(x, y, color=colors[cluster_count], marker='o', linestyle='None')
        #plt.plot(data[medoid][0], data[medoid][1], color='k', marker='o', linestyle='None')
        cluster_count += 1
    plt.savefig('clusters.png')

if __name__ == "__main__":
    main()
