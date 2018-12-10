
# coding: utf-8

# Imports

# In[1]:

import snap
import os
import json
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import itertools


# Constants

# In[27]:

motif_counts_file = "motif_counts_final.txt"
edge_list_dir = "edge_lists_0.25"
edge_list_file_type = "_edge_list_0.25_threshold.txt"
feature_vector_file = "networkname_features.txt"
labels_file = "ADHD200_CC200_labels.csv"
labels_to_int = {"ADHD-Combined":0, 'ADHD-Inattentive':1, 'ADHD-Hyperactive/Impulsive':2, 'Typically Developing':3}


# In[433]:

# load motif_counts a dictionary from file name to 3-node motif counts
motif_counts_json = open(motif_counts_file)
json_str = motif_counts_json.read()
motif_counts = json.loads(json_str)


# In[28]:

# get the features
network_to_features = {}
count = 0
for filename in os.listdir(edge_list_dir):
    if filename.endswith(edge_list_file_type): 
        Graph = snap.LoadEdgeList(snap.PNGraph, edge_list_dir+"/"+filename, 0, 1)
        UGraph = snap.LoadEdgeList(snap.PUNGraph, edge_list_dir+"/"+filename, 0, 1)
        for node in range(190):
            if (Graph.IsNode(node) == False):
                Graph.AddNode(node)
                UGraph.AddNode(node)
        network_name = filename.replace(edge_list_file_type, "")
        CmtyV = snap.TCnComV()
       # feature_vec = list(motif_counts[network_name])
        feature_vec = []
#         DegToCCfV = snap.TFltPrV()
#         result = snap.GetClustCfAll(UGraph, DegToCCfV)
#         feature_vec.extend(result)
#         result = snap.GetBfsEffDiamAll(UGraph, 25, False)
#         feature_vec.extend(result)
#         avg_efficiency = getAvgEfficiency(Graph)
#         feature_vec.append(avg_efficiency)
     #   node_efficiency = getNodeEfficiency(Graph)
    #    feature_vec.extend(node_efficiency)
#         global_efficiency = getGlobalEfficiency(Graph)
#         feature_vec.append(global_efficiency)
#         GraphClustCoeff = snap.GetClustCf (Graph, -1)
#         feature_vec.append(GraphClustCoeff)
#         feature_vec.append(GraphClustCoeff/result[2])
#         betweenness_centrality = getBetweennessCentrality(Graph)
   #     feature_vec.extend(betweenness_centrality)
        if count % 50 == 0:
            print(count)
# ABOVE THIS LINE IS HELPFUL
        betweenness_centrality = getBetweennessCentrality(Graph)
        feature_vec.extend(betweenness_centrality)
        clustering_coeffs = getClusteringCoeff(Graph)
        feature_vec.extend(clustering_coeffs)
        closeness_centrality = getClosenessCentrality(Graph)
        feature_vec.extend(closeness_centrality)
        farness_centrality = getFarnessCentrality(Graph)
        feature_vec.extend(farness_centrality)
        eigen_centrality = getEigenVectorCentrality(UGraph)
        feature_vec.extend(eigen_centrality)
        node_efficiency = getNodeEfficiency(Graph)
        feature_vec.extend(node_efficiency)
        network_to_features[network_name] = feature_vec
        count += 1


# In[43]:

# get the labels
labels_df = pd.read_csv(labels_file) 
labels_df = labels_df[['upload_data.network_name','upload_data.subject_pool']]
network_labels = dict([(network,label) for network,label in zip(labels_df['upload_data.network_name'], labels_df['upload_data.subject_pool'])])


# In[44]:

# format data & labels
all_features = []
all_labels = []
all_binary_labels = [] # 0 if no ADHD, 1 otherwise
adhd_features = []
adhd_labels = []
adhd_no_hyperactive_features = []
adhd_no_hyperactive_labels = []
for network in network_to_features:
    features = network_to_features[network]
    network = network.replace(edge_list_file_type, "")
    label = labels_to_int[network_labels[network]]
    all_features.append(features)
    all_labels.append(label)
    if label != labels_to_int['Typically Developing']:
        all_binary_labels.append(1)
        adhd_features.append(features)
        adhd_labels.append(label)
        if label != labels_to_int['ADHD-Hyperactive/Impulsive']:
            adhd_no_hyperactive_features.append(features)
            adhd_no_hyperactive_labels.append(label)
    else:
        all_binary_labels.append(0)


# In[74]:

# actual ML
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression

features_train, features_test, labels_train, labels_test = sklearn.model_selection.train_test_split(adhd_no_hyperactive_features,adhd_no_hyperactive_labels, test_size=0.2)
# lin_clf = RandomForestClassifier(n_estimators=40)
lin_clf = GaussianNB()
# lin_clf = LogisticRegression()
# lin_clf = svm.SVC()
lin_clf.fit(features_train, labels_train) 
labels_predicted = lin_clf.predict(features_test)  
sklearn.metrics.accuracy_score(labels_test, labels_predicted)


# In[75]:

# confusion matrix
plt.figure()
# classes = ["ADHD-Combined", 'ADHD-Inattentive', 'ADHD-Hyperactive/Impulsive', 'Typically Developing']
# classes = ['Typically Developing', 'ADHD']
classes = ["ADHD-Combined", 'ADHD-Inattentive']
cm = sklearn.metrics.confusion_matrix(labels_test, labels_predicted)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2
fmt = 'd'
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[307]:

# look at the values of the features
feature_names = ["avg clustering coeff", "# closed triads", "# open triads", "eff diameter of graph",
                "eff diameter of graph", "diameter", "avg shortest path length", "avg efficiency"]
for i in range(190):
    feature_names.append("node efficiency for node "+str(i))
feature_names.extend(["global efficiency", "graph clustering coeff", "small worldness"])
for i in range(190):
    feature_names.append("betweeness centrality for node "+str(i))
class_labels = lin_clf.classes_
topn_class1 = sorted(zip(lin_clf.feature_count_[0], feature_names),reverse=True)[:11]
topn_class2 = sorted(zip(lin_clf.feature_count_[1], feature_names),reverse=True)[:11]
print("Important values in Typical Developing")
for coef, feat in topn_class1:
    print(class_labels[0], coef, feat)
print("-----------------------------------------")
print("Important values in ADHD")
for coef, feat in topn_class2:
    print(class_labels[1], coef, feat) 


# In[59]:

# distribution of characteristic path length
# feature_names = ["avg clustering coeff", "# closed triads", "# open triads", "eff diameter of graph",
                #"eff diameter of graph", "diameter", "avg shortest path length", "avg efficiency"]
# feature_names.extend(["global efficiency", "graph clustering coeff", "small worldness"])
adhd_cpl = []
normal_cpl = []
combined = []
inattentive = []
for i in range(len(adhd_no_hyperactive_labels)):
    feature_vec = adhd_no_hyperactive_features[i]
    label = adhd_no_hyperactive_labels[i]
    if label == 0:
        combined.append(feature_vec[20])
    else:
        inattentive.append(feature_vec[20])
        
        
# char path length = 6


# In[60]:

plt.hist(combined, bins=20, alpha=0.5, label='ADHD-Combined')
plt.hist(inattentive, bins=20, alpha=0.5, label='ADHD-Inattentive')
plt.xlabel("Node Efficiency")
plt.legend()
plt.show()


# In[4]:

def getCommunities(CmtyV, numNodes):
    community_matrix = np.zeros((numNodes, numNodes))
    node_calCommunity = {}
    for Cmty in CmtyV:
        for NI in Cmty:
            node_calCommunity[NI] = Cmty
    for i in range(numNodes):
        for j in range(i, numNodes):
            if node_calCommunity[i] == node_calCommunity[j]:
                community_matrix[i][j] = 1 # indicates i and j in same community
    # now just get data points in upper triangular part of matrix as will only get 1s there
    iu1 = np.triu_indices(numNodes)
    return community_matrix[iu1]


# In[5]:

def getClusteringCoeff(Graph):
    coeffs = []
    for i in range(Graph.GetNodes()):
        coeffs.append(snap.GetNodeClustCf(Graph, i))
    return coeffs


# In[6]:

def getBetweennessCentrality(Graph):
    centrality = []
    Nodes = snap.TIntFltH()
    Edges = snap.TIntPrFltH()
    snap.GetBetweennessCentr(Graph, Nodes, Edges, 1.0, True)
    nodeid_centrality = {}
    for node in Nodes:
        nodeid_centrality[node] = Nodes[node]
    for node_id in sorted(nodeid_centrality):
        centrality.append(nodeid_centrality[node_id])
    return centrality


# In[7]:

def getPageRank(Graph):
    nodeid_prank = {}
    prank = []
    PRankH = snap.TIntFltH()
    snap.GetPageRank(Graph, PRankH)
    for node in PRankH:
        nodeid_prank[node] = PRankH[node]
    for node_id in sorted(nodeid_prank):
        prank.append(nodeid_prank[node_id])
    return prank


# In[8]:

def getClosenessCentrality(Graph):
    coeffs = []
    for i in range(Graph.GetNodes()):
        coeffs.append(snap.GetClosenessCentr(Graph, i))
    return coeffs


# In[9]:

def getFarnessCentrality(Graph):
    coeffs = []
    for i in range(Graph.GetNodes()):
        coeffs.append(snap.GetClosenessCentr(Graph, i))
    return coeffs


# In[10]:

def getEigenVectorCentrality(Graph):
    centrality = []
    NIdEigenH = snap.TIntFltH()
    snap.GetEigenVectorCentr(Graph, NIdEigenH)
    nodeid_centrality = {}
    for node in NIdEigenH:
        nodeid_centrality[node] = NIdEigenH[node]
    for node_id in sorted(nodeid_centrality):
        centrality.append(nodeid_centrality[node_id])
    return centrality


# In[11]:

def getNodeEfficiency(Graph): 
    nodes = [node.GetId() for node in Graph.Nodes()]
    efficiency = []
    n = len(nodes)
    for i in range(0, n): 
        for j in range(i + 1, n): 
            if i != j: 
                efficiency.append(1 / float(snap.GetShortPath(Graph, nodes[i], nodes[j])))    
    return efficiency


# In[12]:

def getAvgEfficiency(Graph): 
    nodes = [node.GetId() for node in Graph.Nodes()]
    efficiency = 0 
    n = len(nodes)
    for i in range(0, n): 
        for j in range(i + 1, n): 
            if i != j: 
                efficiency += 1 / float(snap.GetShortPath(Graph, nodes[i], nodes[j]))    
    return 1 / float( n * n-1 ) * efficiency


# In[13]:

def getGlobalEfficiency(Graph): 
    E_G = getAvgEfficiency(Graph)
    num_nodes = Graph.GetNodes()
    G_ideal = snap.GenFull(snap.PUNGraph, num_nodes)
    E_G_ideal = getAvgEfficiency(G_ideal)
    return E_G / float(E_G_ideal)


# In[4]:

import matplotlib.pyplot as plt
# threshold graphs
thresholds = [0, 0.05, .1, .15, .2, .25, .3, .35, .4, .45, .5]
adhd_type_acc = [0.61, 0.51, 0.55, 0.58, 0.59, 0.67, 0.56, 0.59, 0.45, 0.48, 0.46]
adhd_pres_acc = [0.56, 0.53, 0.62, 0.58, 0.59, 0.63, 0.63, 0.65, 0.68, 0.63, 0.59]

plt.plot(thresholds, adhd_type_acc, 'r--', label="ADHD Type")
plt.plot(thresholds, adhd_pres_acc, 'b--', label = "ADHD or Normal Developing")
plt.xlabel("Threshold Value")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[ ]:



