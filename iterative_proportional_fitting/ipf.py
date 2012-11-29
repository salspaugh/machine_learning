#!/usr/bin/env python

import math
import numpy as np

from collections import defaultdict

ROWS = 0
COLS = 1

class Graph(object):

    def __init__(self, nodes, cliques, data, debug=True):
        """
        An object to represent an undirected graph of variables `nodes` and 
        potentials on `cliques` of these nodes for the purpose of running 
        iterative proportional fit given a set of samples `data`.
        Assumes variables (nodes) are binary.
        """
        self.nodes = nodes
        self.cliques = cliques
        
        self.configurations = {}
        self.empirical_marginals = {}
        self.estimated_marginals = {}
        self.potentials = {}
        self.joints = {}
        self.configurations[nodes] = self.build_configuration_table(nodes)
        for clique in self.cliques:
            self.configurations[clique] = self.build_configuration_table(clique)
            self.empirical_marginals[clique] = np.matrix([[0.]]*2**len(clique))
            self.estimated_marginals[clique] = np.matrix([[0.]]*2**len(clique))
            self.potentials[clique] = np.matrix([[1.]]*2**len(clique))
        
        self.data = data
        self.number_of_samples = self.data.shape[COLS]

        self.debug = debug

    def build_configuration_table(self, variables):
        """
        Build a list of column vectors (actually, numpy matrices) such that 
        each entry in the list is a `variable`-length vector that represents a 
        `variable`-length number that is one larger than the previous one in
        list. In other words, the list is a set of binary numbers of length 
        `variable` that count from 0 to 2**length(`variable`). Return the list.
        Assumes the possible states a variable can be in is {0,1}.
        """
        old_configuration = np.matrix([[0.]]*len(variables))
        configurations = []
        configurations.append(old_configuration)
        for i in xrange(2**len(variables)-1):
            curr_configuration = self.inc(old_configuration)
            configurations.append(curr_configuration)
            old_configuration = curr_configuration
        return configurations

    def inc(self, zero_one_matrix):
        """
        Increment the binary number represented by the column vector (actually,
        numpy matrix) `zero_one_matrix` and return that new value in a new 
        column vector.
        Assumes the possible states a variable can be in is {0,1}.
        """
        new_matrix = np.matrix(zero_one_matrix, copy=True)
        carry = 1
        for i in reversed(range(zero_one_matrix.shape[ROWS])):
            sum = int(zero_one_matrix[i,0]) + carry
            bit = sum % 2
            carry = sum / 2
            new_matrix[i,0] = bit
        return new_matrix

    def count(self, clique, entry):
        """ 
        Count the number of instances in the data where the nodes in the given
        clique have values equal to the given configuration entry.
        
        Assumes that each column of the data represents a new sample of the
        variables represented by all nodes {X_1,...X_d} so that if there are n
        samples then data is a dxn matrix.
        """
        count = 0.
        for sample in range(self.number_of_samples):
            if (self.data[clique,sample] == entry).all():
                count += 1.
        return count

    def compute_empirical_marginals(self):
        """
        Compute the empirical marginals of a certain graph with nodes {X_1,...,X_d}
        given data from n samples of these nodes i.e. a dxn matrix of data.
        """
        normalization_factor = self.number_of_samples
        for clique in self.cliques:
            row = 0
            for entry in self.configurations[clique]:
                self.empirical_marginals[clique][row,0] = self.count(clique, entry) 
                row += 1                                        # Laplace smoothing:
            if (self.empirical_marginals[clique] == 0.).any(): # if any entries are zero
                self.empirical_marginals[clique] += 1.          # add one to all counts
            self.empirical_marginals[clique] /= normalization_factor

    def matrix_column_to_string(self, matrix, col):
        s = ''
        for row in range(matrix.shape[ROWS]):
            s = ''.join([s, str(int(matrix[row,col]))])
        return s

    def potential_products(self, configuration):
        """
        Multiplies the clique potentials together at the d-dimensional point 
        `configuration`.
        """
        product = 1.
        for clique in self.cliques:
            these_vars = configuration[clique,0]
            idx = int(self.matrix_column_to_string(these_vars, 0), 2)
            product = product * math.exp(self.potentials[clique][idx])
        return product

    def joint(self, configuration):
        """
        Computes the joint at the d-dimensional point `configuration`. The
        joint is the normalized product of potentials at `configuration`.
        """
        normalization_factor = 0.
        for entry in self.configurations[self.nodes]:
            normalization_factor += self.potential_products(entry)
        self.joints[configuration] = self.potential_products(configuration) / normalization_factor
        return self.joints[configuration]

    def estimated_marginal(self, clique, entry):
        """
        Computes the marginal of `clique` at the size(clique)-dimensional
        point `entry` by summing over all joint configurations where the
        given clique points of that configuration are equal to `entry`.
        """
        sum = 0.
        for configuration in self.configurations[self.nodes]:
            if (configuration[clique,0] == entry).all():
                sum += self.joint(configuration)
        return sum

    def update_estimated_marginals(self):
        """
        Updates all of the clique marginals based on the current values of the
        potential functions.
        """
        for clique in self.cliques:
            row = 0
            for entry in self.configurations[clique]:
                self.estimated_marginals[clique][row,0] = self.estimated_marginal(clique, entry)
                row += 1

    def log_empirical_over_estimated_marginal(self, clique):
        """
        Returns the log of the empirical marginal for `clique` over the
        estimated marginal for the `clique`.
        """
        if self.debug:
            print "Taking log of: "
            self.print_functions_of_clique(self.empirical_marginals[clique], clique, 'muhat')
            self.print_functions_of_clique(self.estimated_marginals[clique], clique, 'mutilde')
        return np.log(self.empirical_marginals[clique] / self.estimated_marginals[clique])

    def do_IPF_update(self):
        """
        Does one iteration of iterative proportional fitting. For each clique 
        potential, it computes the new clique potential at this timestep based
        on that at the last timestep and the IPF update rule, also updating 
        the estimated marginals each time a clique potential is updated.
        """
        for clique in self.cliques:
            self.update_estimated_marginals()
            self.potentials[clique] = self.potentials[clique] + \
                                        self.log_empirical_over_estimated_marginal(clique)
    
    def do_IPF(self, iterations):
        """
        Main loop for iterative proportional fitting. Does `iteration` number
        of iterations, updating the graph potentials and marginals as it goes.
        """
        for t in xrange(iterations):
            if self.debug:
                print "Iteration %d" % t
                self.print_potentials()
            self.do_IPF_update()

    def compute_log_likelihood(self):
        self.log_likelihood = 0. 
        for sample in xrange(self.data.shape[COLS]):
            self.log_likelihood += np.log(self.joint(self.data[:,sample]))

    def compute_singleton_node_entropy(self, node):
        singleton = np.matrix([[0.]]*2)
        row = 0
        for entry in [(0),(1)]:
            singleton[row,0] = self.estimated_marginal((node), entry)
            row += 1
        return singleton * np.log(singleton)

    def compute_joint_edge_entropy(self, edge):
        joint = np.matrix([[0.]]*4)
        row = 0
        for entry in [(0,0),(0,1),(1,0),(1,1)]:
            joint[row,0] = self.estimated_marginal(edge, entry)
            row += 1
        return joint * np.log(joint)

    def compute_mutual_information(self, edges):
        trees = [[(0,1),(1,3),(2,3)],
            [(0,2),(1,3),(2,3)],
            [(0,1),(0,2),(2,3)],
            [(0,1),(0,2),(1,3)],
            [(0,1),(1,2),(1,3)],
            [(0,1),(0,3),(0,2)],
            [(0,2),(1,2),(2,3)],
            [(0,3),(1,3),(2,3)],
        ]
        self.mutual_informatons = {}
        self.max_mutual_information = 0.
        for tree in trees:
            mutual_information = 0.
            for edge in tree:
                mutual_information += self.compute_joint_edge_entropy(edge)
            for node in self.nodes:
                mutual_information += self.compute_singleton_node_entropy(node)
            if mutual_information > self.max_mutual_information:
                self.max_mutual_information = mutual_information
                self.max_mutual_information_tree = tree
            self.mutual_information[tree] = mutual_information

    def print_functions_of_clique(self, function, clique, fnname):
        print "\tclique: %s" % str(clique)
        for entry in self.configurations[clique]:
            input = self.matrix_column_to_string(entry, 0)
            output = '{0:2.2f}'.format(function[int(input, 2),0])
            s = '\t\t' + fnname + '(' + input + ') = ' + output 
            print s

    def print_functions(self, functions, fnname):
        for (clique, function) in functions.iteritems():
            self.print_functions_of_clique(function, clique, fnname)

    def print_empirical_marginals(self):
        self.print_functions(self.empirical_marginals, 'mu')

    def print_potentials(self):
        self.print_functions(self.potentials, 'theta')
   
    def print_joint(self): # different from other functions --- just a map FIXME
        for entry in self.configurations[self.nodes]:
            input = self.matrix_column_to_string(entry, 0)
            s = '\t\tP(' + input + ') = ' + str(self.joints[entry]) 
            print s

    def print_log_likelihood(self):
        print "\tL(...) =", self.log_likelihood

    def print_configurations(self):
        for (variables, configuration) in self.configurations.iteritems():
            print "variables: %s" % str(variables)
            for entry in configuration:
                print self.matrix_column_to_string(entry, 0)

    def print_data(self):
        np.set_printoptions(linewidth=150)
        print self.data

    def print_results(self):
        print "Final empirical marginals:"
        self.print_empirical_marginals()
        print "Final potentials:"
        self.print_potentials()
        print "Final joint:"
        self.print_joint()
        print "Final log likelihood:"
        self.print_log_likelihood()
       
    def iterative_proportional_fit(self):
        self.compute_empirical_marginals()
        self.do_IPF(100)
        self.compute_log_likelihood()
        self.print_results()
        self.compute_mutual_information()
        print self.max_mutual_information_tree
        print self.max_mutual_information

def ps5q3_graph(cliques, debug=False):
    nodes = (0,1,2,3)
    data = []
    with open("Pairwise.dat") as datafile:
        rows = [line.split() for line in datafile.readlines()]
        data = [[float(element) for element in row] for row in rows]  
    data = np.matrix(data)
    return Graph(nodes, cliques, data, debug=debug)

def solve_ps5q3_part(part, cliques, debug=False):
    print "Solution to part %s " % str(part)
    g = ps5q3_graph(cliques, debug=debug)
    g.iterative_proportional_fit()

def solve_ps5q3(debug=False):

    cliques_part1 = [(0,1), (0,3), (1,2), (2,3)]
    solve_ps5q3_part("1", cliques_part1)
    
    #cliques_part2 = [(0,1,2), (0,3)]
    cliques_part2 = [(0,1), (0,2), (0,3), (1,2)]
    solve_ps5q3_part("2", cliques_part2)
    
    #cliques_part3 = [(0,1,2,3)]
    cliques_part3 = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    solve_ps5q3_part("3", cliques_part3)


if __name__ == '__main__':
    """Program entry point and option parsing.
    Options are to turn on debugging and to provide an arbitrary graph 
    instance and data to compute Iterative Proportional Fitting over. 
    Program assumes that the graph is provided in JSON form with the nodes, 
    cliques, and data provided. Program assumes nodes represent binary
    variables.
    """

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename",
                      help="write report to FILE", metavar="FILE")
    parser.add_option("-d", "--debug", dest="debug",
                      action="store_true", default=False,
                      help="output debugging messages", metavar="DEBUG")

    (options, args) = parser.parse_args()

    if options.filename is not None:
        import json
        json_data = open(options.filename).read()
        data = json.loads(json_data)
        graph_data = data['graph']
        graph = Graph(int(graph_data['nodes']), 
                        graph_data['cliques'],
                        graph_data['data'],
                        debug=graph_data['debug'])
        graph.iterative_proportional_fit()
    else:
        solve_ps5q3(debug=options.debug)
    
