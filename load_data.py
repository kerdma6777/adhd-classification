
# coding: utf-8

# Imports

# In[2]:

import os
import numpy as np


# Constants

# In[1]:

# assumes that this code is parallel to the data directory
data_directory = "data_ADHD200_CC200"
edge_list_dir = "edge_lists_0.25"
include_edge_weight = True
edge_weight_threshold = 0.25


# In[4]:

for filename in os.listdir(data_directory):
    if filename.endswith("_connectivity_matrix_file.txt"): 
        adj_matrix = np.loadtxt(data_directory+"/"+filename)
        edge_list = generate_edge_list(adj_matrix)
        network_name = filename.replace("_connectivity_matrix_file.txt", "")
        with open(edge_list_dir+"/"+network_name+'_edge_list_0.25_threshold.txt', 'w+') as f:
            for edge in edge_list:
                f.write("%s\n" % edge)


# In[3]:

def generate_edge_list(adj_matrix):
    edges = []
    for r in range(len(adj_matrix)):
        for c in range(len(adj_matrix[0])):
            if (adj_matrix[r][c] > edge_weight_threshold):
                edge_str = str(r)+"\t"+ str(c)
                if (include_edge_weight):
                    edge_str += "\t" + str(adj_matrix[r][c])
                edges.append(edge_str)
    return edges


# In[ ]:




# In[ ]:



