#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import timeit

from itertools import product
import timeit
import math
import pandas as pd



sequences = np.load("/alina-data1/prakash/Anderson/data/embeddings/t_Cell/t_cell_PseACC_seq.npy",allow_pickle=True)
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
# spaced_kmer_length = 3 # 6 , 4
# 
# Kmer = spaced_kmer_length
# 
# unique_seq_kmers_final_list = [''.join(c) for c in product('ACDEFGHIKLMNPQRSTVWXY-', repeat=spaced_kmer_length)]  
# 
# 
# start = timeit.default_timer()
# 
# frequency_vector = []
# 
# for seq_ind in range(len(seq_val_global)):
#     print("index: ",seq_ind,"/",len(seq_val_global))
#     se_temp = seq_val_global[seq_ind]
#     
#     k_mers_final = build_kmers(se_temp,spaced_kmer_length)
#     
#     #create dictionary
#     idx = pd.Index(k_mers_final) # creates an index which allows counting the entries easily
#     # print('Here are all of the viral species in the dataset: \n', len(idx),"entries in total")
#     aq = idx.value_counts()
#     counter_tmp = aq.values
#     gmers_tmp = aq.index
#     # counter_tmp,gmers_tmp
# 
# 
#     #create frequency vector
#     #cnt_check2 = 0
#     listofzeros = [0] * len(unique_seq_kmers_final_list)
#     for ii in range(len(gmers_tmp)):
#         seq_tmp = gmers_tmp[ii]
#     #     listofzeros = [0] * len(unique_seq_kmers_final_list)
#     #     for j in range(len(seq_tmp)):
#         ind_tmp = unique_seq_kmers_final_list.index(seq_tmp)
#         listofzeros[ind_tmp] = counter_tmp[ii]
#     frequency_vector.append(listofzeros)
# 

# In[19]:


# np.save("/olga-data1/prakash/PDB_ICLR/Embeddings/Spike2Vec_Embedding_480.npy",frequency_vector)


def build_minimizer(sequence, ksize, m_size):
    # https://homolog.us/blogs/bioinfo/2017/10/25/intro-minimizer/
#     seq="ATGCGATATCGTAGGCGTCGATGGAGAGCTAGATCGATCGATCTAAATCCCGATCGATTCCGAGCGCGATCAAAGCGCGATAGGCTAGCTAAAGCTAGCA"
#     sequence = seq[:]

#     asd = str(sequence)
#     asd = np.array2string(sequence)
    
    string_parsing = []
    for ind_test in range(len(sequence)):
        string_parsing.append(str(sequence[ind_test]))
    
    asd = str(string_parsing)
    aa_lst_1 = asd.replace(",","")
    aa_lst_2 = aa_lst_1.replace("[","")
    aa_lst_3 = aa_lst_2.replace("\"","")
    aa_lst_4 = aa_lst_3.replace("]","")
    aa_lst_5 = aa_lst_4.replace("'","")
    aa_lst_6 = aa_lst_5.replace(" ","")
    aa_lst_6

#     print(aa_lst_6)
    seq = aa_lst_6[:]
    rev=seq[::-1]
    
    Kmer=ksize
    M=m_size
    L=len(seq)

    minimizers = []
    k_mers_final = []
    for i in range(0, L-Kmer+1):

            sub_f=seq[i:i+Kmer]
            sub_r=rev[L-Kmer-i:L-i]

            min="ZZZZZZZZZZZZZ"
            for j in range(0, Kmer-M+1):
                    sub2=sub_f[j:j+M]
                    if sub2 < min:
                            min=sub2
                    sub2=sub_r[j:j+M]
                    if sub2 < min:
                            min=sub2
            minimizers.append(min)
            k_mers_final.append(sub_f)
#             print(sub_f,min)

#     print("unique minimizers = ",len(np.unique(minimizers)))
#     print("unique kmers = ",len(np.unique(k_mers_final)))

    return minimizers


start = timeit.default_timer()


k_size_val = 9
m_size_val = 3

protein_kmers_list = []
for protein_kmers in range(len(sequences)):
#     print(protein_kmers, "/",len(sequences))
    k_mers_vals = build_minimizer(sequences[protein_kmers],k_size_val, m_size_val)

    # str(k_mers_vals[0])
    k_mers_list = []
    for mers_ind in range(len(k_mers_vals)):
        k_mers_list.append(str(k_mers_vals[mers_ind]))
        
    protein_kmers_list.append(k_mers_list)


seq_kmers_final = []
for i in range(len(protein_kmers_list)):
    tmp = protein_kmers_list[i]
    tmp_seq = []
    for j in range(len(protein_kmers_list[i])):
        aa = tmp[j]
        aa_lst = str(list(aa))
        aa_lst_1 = aa_lst.replace(",","")
        aa_lst_2 = aa_lst_1.replace("[","")
        aa_lst_3 = aa_lst_2.replace("\"","")
        aa_lst_4 = aa_lst_3.replace("]","")
        aa_lst_5 = aa_lst_4.replace("'","")
        aa_lst_6 = aa_lst_5.replace(" ","")
        tmp_seq.append(aa_lst_6)
    seq_kmers_final.append(tmp_seq)

 
unique_seq_kmers_final_list = [''.join(c) for c in product('ACDEFGHIKLMNPQRSTVWXY-', repeat=3)]   
 
frequency_vector = []
#cnt_check2 = 0
for ii in range(len(seq_kmers_final)):
    seq_tmp = seq_kmers_final[ii]
    listofzeros = [0] * len(unique_seq_kmers_final_list)
    for j in range(len(seq_tmp)):
        ind_tmp = unique_seq_kmers_final_list.index(seq_tmp[j])
        listofzeros[ind_tmp] = listofzeros[ind_tmp] + 1
    frequency_vector.append(listofzeros)

stop = timeit.default_timer()
print("Time for minimizer ", stop - start) 
    
    
from numpy import asarray
from numpy import save

data = np.array(frequency_vector)
np.save("/alina-data1/prakash/Anderson/data/embeddings/t_Cell/Minimizer_Embedding_t_cell.npy",frequency_vector)

print("Embedding Saved!!")

print("All Processing Done!!!")




