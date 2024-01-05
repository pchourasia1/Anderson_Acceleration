#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import timeit

from itertools import product
import timeit
import math
import pandas as pd


# /olga-data1/Sarwan/PDB2Vec/Embeddings/STCRDAB_Sequences_From_PDB_480.npy
# /olga-data1/Sarwan/PDB2Vec/Embeddings/PDB_Bind_Preprocessed_Sequences_From_PDB_Bind_3792.npy

seq_val_global = np.load("/olga-data1/Sarwan/PDB2Vec/Embeddings/PDB_Bind_Preprocessed_Sequences_From_PDB_Bind_3792.npy",allow_pickle=True)


print("Data Loaded!!")


def build_kmers(sequence, ksize):
    kmers = []
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)

    return kmers


# In[18]:



# gmer_length = 7 # 9 , 5
spaced_kmer_length = 3 # 6 , 4

Kmer = spaced_kmer_length

unique_seq_kmers_final_list = [''.join(c) for c in product('ACDEFGHIKLMNPQRSTVWXY-', repeat=spaced_kmer_length)]  


start = timeit.default_timer()

frequency_vector = []

for seq_ind in range(len(seq_val_global)):
    print("index: ",seq_ind,"/",len(seq_val_global))
    se_temp = seq_val_global[seq_ind]
    
    k_mers_final = build_kmers(se_temp,spaced_kmer_length)
    
    #create dictionary
    idx = pd.Index(k_mers_final) # creates an index which allows counting the entries easily
    # print('Here are all of the viral species in the dataset: \n', len(idx),"entries in total")
    aq = idx.value_counts()
    counter_tmp = aq.values
    gmers_tmp = aq.index
    # counter_tmp,gmers_tmp


    #create frequency vector
    #cnt_check2 = 0
    listofzeros = [0] * len(unique_seq_kmers_final_list)
    for ii in range(len(gmers_tmp)):
        seq_tmp = gmers_tmp[ii]
    #     listofzeros = [0] * len(unique_seq_kmers_final_list)
    #     for j in range(len(seq_tmp)):
        ind_tmp = unique_seq_kmers_final_list.index(seq_tmp)
        listofzeros[ind_tmp] = counter_tmp[ii]
    frequency_vector.append(listofzeros)


# In[19]:


np.save("/olga-data1/prakash/PDB_ICLR/Embeddings/Spike2Vec_Embedding_PDB_Bind_3792.npy",frequency_vector)

print("Embedding Saved!!")


print("All Processing Done!!!")




