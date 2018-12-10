
# coding: utf-8

# Imports

# In[1]:

import snap
import os
from itertools import permutations
import json


# Constants

# In[8]:

# assumes that this code is parallel to the data directory
edge_list_dir = "edge_lists_ADHD200_CC200"
edge_list_file_type = "_edge_list_no_weight.txt"
directed_3 = load_3_subgraphs()
motif_file = "motif_counts.txt"


# In[11]:

index = 0
network_to_motif = {}
for filename in os.listdir(edge_list_dir):
    if filename.endswith(edge_list_file_type): 
        Graph = snap.LoadEdgeList(snap.PNGraph, edge_list_dir+"/"+filename, 0, 1)
        network_name = filename.replace("edge_list_file_type", "")
        enumerate_subgraph(Graph)
        network_to_motif[network_name] = list(motif_counts)
        index += 1
        if index % 10 == 0:
            print(index)
        if index % 50 == 0:
            with open(motif_file, 'w+') as m_file:
                m_file.write(json.dumps(network_to_motif))


# In[12]:

with open("motif_counts_final.txt", 'w+') as m_file:
       m_file.write(json.dumps(network_to_motif))


# Code from HW2

# In[3]:

def enumerate_subgraph(G, k=3, verbose=False):
    '''
    This is the main function of the ESU algorithm.
    Here, you should iterate over all nodes in the graph,
    find their neighbors with ID greater than the current node
    and issue the recursive call to extend_subgraph in each iteration

    A good idea would be to print a progress report on the cycle over nodes,
    So you get an idea of how long the algorithm needs to run
    '''
    global motif_counts
    motif_counts = [0]*len(directed_3) # Reset the motif counts (Do not remove)
    ##########################################################################
    node_itr = G.BegNI()
    count = 0
    num_nodes = G.GetNodes()
    for i in range(num_nodes):
        node_id = node_itr.GetId()
        neighbors = node_itr.GetDeg()
        v_ext = set()
        for j in range(neighbors):
                neighbor_id = node_itr.GetNbrNId(j)
                if neighbor_id > node_id:
                        v_ext.add(neighbor_id)
        sg = set()
        sg.add(node_id)
        extend_subgraph(G, k, sg, v_ext, node_id, True)
        node_itr.Next()


# In[4]:

def extend_subgraph(G, k, sg, v_ext, node_id, verbose=False):
    '''
    This is the recursive function in the ESU algorithm
    The base case is already implemented and calls count_iso. You should not
    need to modify this.

    Implement the recursive case.
    '''
    # Base case (you should not need to modify this):
    if len(sg) is k:
        count_iso(G, sg, verbose)
        return
    # Recursive step:
    ##########################################################################
    v_ext_orig = v_ext.copy()
    while (len(v_ext) > 0):
        node_id2 = v_ext.pop()
        node_itr = G.GetNI(node_id2)
        neighbors = node_itr.GetDeg()
        v_ext_new = v_ext.copy()
        sg_new = sg.copy()
        sg_new.add(node_id2)
        for i in range(neighbors):
                neighbor_id = node_itr.GetNbrNId(i)
                if neighbor_id > node_id and neighbor_id not in sg_new and neighbor_id not in v_ext_orig:
                        v_ext_new.add(neighbor_id)
        extend_subgraph(G, k, sg_new, v_ext_new, node_id)


# In[5]:

def count_iso(G, sg, verbose=False):
    '''
    Given a set of 3 node indices in sg, obtains the subgraph from the
    original graph and renumbers the nodes from 0 to 2.
    It then matches this graph with one of the 13 graphs in
    directed_3.
    When it finds a match, it increments the motif_counts by 1 in the relevant
    index

    IMPORTANT: counts are stored in global motif_counts variable.
    It is reset at the beginning of the enumerate_subgraph method.
    '''
    if verbose:
        print(sg)
    nodes = snap.TIntV()
    for NId in sg:
        nodes.Add(NId)
    # This call requires latest version of snap (4.1.0)
    SG = snap.GetSubGraphRenumber(G, nodes)
    for i in range(len(directed_3)):
        if match(directed_3[i], SG):
            motif_counts[i] += 1


# In[6]:

def match(G1, G2):
    '''
    This function compares two graphs of size 3 (number of nodes)
    and checks if they are isomorphic.
    It returns a boolean indicating whether or not they are isomorphic
    You should not need to modify it, but it is also not very elegant...
    '''
    if G1.GetEdges() > G2.GetEdges():
        G = G1
        H = G2
    else:
        G = G2
        H = G1
    # Only checks 6 permutations, since k = 3
    for p in permutations(range(3)):
        edge = G.BegEI()
        matches = True
        while edge < G.EndEI():
            if not H.IsEdge(p[edge.GetSrcNId()], p[edge.GetDstNId()]):
                matches = False
                break
            edge.Next()
        if matches:
            break
    return matches


# In[7]:

def load_3_subgraphs():
    '''
    Loads a list of all 13 directed 3-subgraphs.
    The list is in the same order as the figure in the HW pdf, but it is
    zero-indexed
    '''
    return [snap.LoadEdgeList(snap.PNGraph, "./subgraphs/{}.txt".format(i), 0, 1) for i in range(13)]


# In[ ]:



