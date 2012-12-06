#!/usr/bin/env python

import em
import kmedoids
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import spectral

from collections import defaultdict
from splparser.parser import parse as splparse
from splparser.parser import SPLSyntaxError
from zss.compare import distance as tree_dist

TEST_FILE = "points.dat"
TEST_POINTS = 1000

def main():
    
    np.set_printoptions(linewidth=200, threshold=1000000)

    (options, args) = parse_args()

    k = (4 if options.k is None else options.k)
    f = (TEST_FILE if options.file is None else options.file)
    i = options.intermediate_in
    o = options.intermediate_out

    em_gaussian = (options.em and options.euclidean)
    em_multinomial = (options.em and not options.euclidean)

    clusterer_selected = False
    
    if em_multinomial:
        clusterer_selected = True
        clusterer = em.cluster
        pdf = em.multinomial_pdf
        param_init = em.multinomial_parameter_init
        param_update = em.multinomial_parameter_update
    
    if em_gaussian:
        enforce_single_selection(clusterer_selected)
        clusterer_selected = True
        clusterer = em.cluster
        pdf = em.isotropic_bivariate_normal_pdf
        param_init = em.isotropic_bivariate_normal_parameter_init
        param_update = em.isotropic_bivariate_normal_parameter_update
    
    if options.spectral:
        enforce_single_selection(clusterer_selected)
        clusterer_selected = True
        clusterer = spectral.cluster
    
    if options.kmedoids:
        enforce_single_selection(clusterer_selected)
        clusterer_selected = True
        clusterer = kmedoids.cluster
    
    if not clusterer_selected:
        print "Please select one clustering method."
        exit()
    
    if em_multinomial:
        counts = None
        if not options.intermediate_in is None:
            counts = get_data(datafile=i, datareader=read_counts_matrix)
        else:
            counts = count_bigrams(f, savefile=o)
        clusters, centers = clusterer(counts, pdf, param_init, param_update, k=k)

    elif em_gaussian:
        data = get_data(datafile=f, datareader=read_points)
        clusters, centers = clusterer(data, pdf, param_init, param_update, k=k)

    elif options.kmedoids or options.spectral:
        r = (read_points if options.euclidean else read_queries)
        data  = get_data(datafile=f, datareader=r)
        if not options.intermediate_in is None:
            distances = get_data(datafile=i, datareader=read_distance_matrix)
        else:     
            d = build_euclidean_distance_matrix if options.euclidean \
                                                else build_tree_edit_distance_matrix
            distances = compute_distances(data, distancer=d, savefile=o) 
        clusters, centers = clusterer(distances, k=k)
        
    if options.euclidean:
        output_point_clusters(data, clusters, centers)
    else:
        output_query_clusters(f, clusters, centers)

def parse_args():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-k", "--num_clusters", dest="k", type="int",
                      help="number of clusters to look for")
    parser.add_option("-f", "--file", dest="file", type="string", metavar="RAW_IN",
                      help="file containing the data to cluster (supported types: splunk queries and x,y coordinates)")
    parser.add_option("-i", "--intermediate_in", dest="intermediate_in", type="string", metavar="INTERMEDIATE_IN",
                      help="file containing intermediate data that has been preprocessed to speed later computations")
    parser.add_option("-o", "--intermediate_out", dest="intermediate_out", type="string", metavar="INTERMEDIATE_OUT",
                      help="file to write intermediate data, to speed later computations")
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
                      help="assume data is SPL queries (defaults to Gaussian Euclidean x,y points)")
    return parser.parse_args()

def enforce_single_selection(clusterer_selected):
    if clusterer_selected: 
        print "Please select one clustering method."
        exit()

def is_npy(filename):
    return (filename[filename.rfind('.'):] == '.npy')

# DELETE
def read_multinomial_sentence_data(datafile):
    with open(datafile) as data:
        for line in data.readline():
            try:
                float(line.split()[0])
                return read_multinomial_counts(datafile)
            except ValueError:
                return count_bigrams(datafile)
                break
            except IndexError:
                continue
    print "Problem parsing data file. Check that file is not empty."
    exit()

def read_counts_matrix(datafile):
    return np.array(np.load(datafile))

def count_bigrams(datafile, savefile=None):
    if savefile is None:
        savefile = 'bigrams.npy'
    word_idxs = get_word_idxs(datafile)
    d = len(word_idxs.keys())
    all_counts = []
    with open(datafile) as data:
        for line in data.readlines():
            counts = np.zeros((d,d))
            stages = line.split('|')
            fragments = [stage.split() for stage in stages]
            words = [fragment[0] for fragment in fragments]
            print words
            first = '^'
            second = words[0]
            print first, second
            fidx = word_idxs[first]
            sidx = word_idxs[second]
            counts[fidx,sidx] += 1.
            for i in range(len(words)-1):
                #fidx = word_idxs[words[i]]
                #sidx = word_idxs[words[i+1]]
                #first = ''.join(['^', words[i]]) if (i == 0) else words[i]
                #second = ''.join([words[i+1], '$']) if (i+1 == len(words)) else words[i+1]
                first = words[i]
                second = words[i+1]
                print first, second
                fidx = word_idxs[first]
                sidx = word_idxs[second]
                counts[fidx,sidx] += 1.
            first = words[len(words)-1]
            second = '$'
            print first, second
            fidx = word_idxs[first]
            sidx = word_idxs[second]
            counts[fidx,sidx] += 1.
            all_counts.append(counts) 
    all_counts = np.array(all_counts)
    np.save(savefile, all_counts)    
    return all_counts

def get_word_idxs(datafile):
    unique_words = defaultdict(int)
    with open(datafile) as data:
        for line in data.readlines():
            stages = line.split('|')
            fragments = [stage.split() for stage in stages]
            words = [fragment[0] for fragment in fragments]
            unique_words['^'] += 1
            unique_words['$'] += 1
            for i in range(len(words)):
                #word = ''.join(['^',words[i]]) if (i == 0) else words[i]
                #word = ''.join([word, '$']) if (i == len(words) - 1) else word
                unique_words[words[i]] += 1
    d = len(unique_words.keys())
    return dict(zip(unique_words.keys(), range(d)))

# DELETE
def read_multinomial_counts(datafile):
    all_counts = []
    with open(datafile) as data:
        for line in data.readlines():
            counts = [float(elem) for elem in line.split()]
            d = math.sqrt(len(counts))
            counts = np.array(counts).reshape(d,d)
            all_counts.append(counts)
    return np.array(all_counts)

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
    if savefile is None:
        savefile = 'distances.npy'
    np.save(savefile, distances)    
    return distances

def output_point_clusters(data, clusters, centers):
    print_results(data, clusters)
    plot_results(data, clusters, centers)

def output_query_clusters(datafile, clusters, centers):
    with open(datafile) as rawdata:
        print_results(rawdata.readlines(), clusters)

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
