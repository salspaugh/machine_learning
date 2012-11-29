#!/usr/bin/env python
# Purpose: Program to compute Sum-Product over arbitrary graphs and also over 
# the graphs given in Problem Set 4.
# Author: Sara Alspaugh
# Date: 14 October 2012

from collections import defaultdict
import numpy as np

DEBUG = False

def debug(msg):
    """Prints debugging messages if the user sets the DEBUG flag."""
    if DEBUG:
        try:
            print(msg)
        except IOError:
            import traceback
            traceback.print_exception(sys.exc_info())

class Graph(object):
    """An object to store the structure and potential functions of the graph
    we're computing Sum-Product over. Also has functions defined on it which ."""
    
    def __init__(self, m, nodes, edges, neighbors, node_potentials, edge_potentials):
        self.m = m
        self.nodes = nodes
        self.edges = edges
        self.neighbors = neighbors
        self.node_potentials = node_potentials
        self.edge_potentials = edge_potentials

    def initialize_messages(self):
        """Initializes all node messages to 1. Also initializes the list of 
        which neighbors each node has sent to / received from."""
        self.messages = {}
        self.neighbors_received = {}
        self.neighbors_sent = {}
        for node in self.nodes:
            self.neighbors_received[node] = set([])
            self.neighbors_sent[node] = set([])
            self.messages[node] = {}
        for (i,j) in self.edges: 
            self.messages[i][j] = np.matrix([[1.]]*self.m, float)
            self.messages[j][i] = np.matrix([[1.]]*self.m, float)

    def collect_messages_from_neighbors(self, node, recipient):
        """Multiplies all messages from a given node's neighbors together except
        the one that is being sent to i.e., the recipient."""
        messages_product = np.matrix([[1.]]*self.m, float)
        for neighbor in self.neighbors[node]:
            if neighbor != recipient:
                messages_product = np.multiply(messages_product, self.messages[neighbor][node])
        return messages_product

    def send_messages(self, to_node, from_node):
        """Computes the message from node from_node to node to_node."""
        debug("Sending message from node %d to node %d" % (from_node, to_node)) 
        shared_edge = (to_node, from_node) if to_node < from_node else (from_node, to_node)
        messages_product = self.collect_messages_from_neighbors(from_node, to_node)
        node_potentials_times_messages = np.multiply(self.node_potentials[from_node], messages_product)
        self.messages[from_node][to_node] = self.edge_potentials[shared_edge] * node_potentials_times_messages
        debug("Message(%d, %d) = \n%s" % (from_node, to_node, str(self.messages[from_node][to_node])))

    def should_send(self, node, neighbor):
        """Returns true if node has received all messages from all nodes except 
        neighbor, meaning that the node can send a message to neighbor now."""
        return (self.neighbors_received[node] >= self.neighbors[node] - set([neighbor]) 
                    and not neighbor in self.neighbors_sent[node])

    def do_iteration(self):
        """Iterates through all nodes and has them send a message if they can 
        send a message this iteration."""
        for node in self.nodes:
            for neighbor in self.neighbors[node]:
                if self.should_send(node, neighbor):
                    self.send_messages(neighbor, node)
                    self.neighbors_received[neighbor].add(node)
                    self.neighbors_sent[node].add(neighbor)
            if self.neighbors_received[node] != self.neighbors[node]:
                self.unconverged = True

    def compute_marginals(self):
        """Computes the marginals once all of the final messages have been 
        received by all nodes."""
        self.marginals = {}
        for node in self.nodes:
            messages_product = np.matrix([[1.]]*self.m, float)
            for neighbor in self.neighbors[node]:
                messages_product = np.multiply(messages_product, self.messages[neighbor][node])
            self.marginals[node] = np.multiply(self.node_potentials[node], messages_product)
            self.marginals[node] = np.divide(self.marginals[node], np.sum(self.marginals[node]))

    def sum_product(self):
        """Runs the Sum-Product algorithm according to the rule that nodes
        only send to a neighbor once they have received all messages from 
        their other neighbors. Nodes send messages until all messages have
        been sent, marking convergence. Lastly, marginals are computed and
        returned."""
        self.initialize_messages()
        self.unconverged = True
        iterations = 0
        while self.unconverged:
            self.unconverged = False
            debug("Iteration %d" % iterations)
            self.do_iteration()
            iterations += 1
        self.compute_marginals()
        return self.marginals

    def print_marginals(self):
        for (node, marginal) in self.marginals.iteritems():
            print("Node %d marginal = \n%s" % (node, str(marginal)))

def ps4q1_graph(a=1., b=.5):
    """Returns an instance of the graph specific to Problem Set 4."""
    nodes = set([1, 2, 3, 4, 5, 6])
    edges = [(1,2), (1,3), (2, 4), (2, 5), (3,6)]
    neighbors = {
        1 : set([2, 3]), 
        2 : set([1, 4, 5]), 
        3 : set([1, 6]), 
        4 : set([2]), 
        5 : set([2]), 
        6 : set([3])}
    node_potentials = {}
    edge_potentials = {}

    for node in nodes:
        if node % 2 == 0:
            #node_potentials[node] = np.matrix([[3., 1., 2.]], float)
            node_potentials[node] = np.matrix([[3.], [1.], [2.]], float)
        else:
            #node_potentials[node] = np.matrix([[1., 2., 3.]], float)
            node_potentials[node] = np.matrix([[1.], [2.], [3.]], float)

    edge_potential = np.matrix([[a, b, b], [b, a, b], [b, b, a]], float)
    for edge in edges:
        edge_potentials[edge] = edge_potential

    g = Graph(3, nodes, edges, neighbors, node_potentials, edge_potentials)
    return g

def solve_ps4q1_inst(a=1., b=.5):
    """Solves an instance of the graph given in Problem Set 4."""
    g = ps4q1_graph(a=a, b=b)
    g.sum_product()
    g.print_marginals()

def solve_ps4q1():
    """Solves both versions of the graph given in Problem Set 4."""
    marginals_part1 = solve_ps4q1_inst()
    marginals_part2 = solve_ps4q1_inst(a=1., b=2.)

if __name__ == '__main__':
    """Program entry point and option parsing.
    Options are to turn on debugging and to provide an arbitrary graph 
    instance to compute Sum-Product over. Program assumes that the graph
    is provided in JSON form with the nodes, edges, neighbors, and 
    potential functions included."""

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename",
                      help="write report to FILE", metavar="FILE")
    parser.add_option("-d", "--debug", dest="debug",
                      action="store_true", default=False,
                      help="output debugging messages", metavar="DEBUG")

    (options, args) = parser.parse_args()

    if options.debug:
        DEBUG = True

    if options.filename is None:
        solve_ps4q1()
    else: # assumes graph information in is JSON
        import json
        json_data = open(options.filename).read()
        data = json.loads(json_data)
        graph_data = data['graph']
        graph = Graph(int(graph_data['m']), 
                        graph_data['nodes'],
                        graph_data['edges'],
                        graph_data['neigbors'],
                        np.matrix(graph_data['node_potentials']),
                        np.matrix(graph_data['edge_potentials']))
        graph.sum_product()
