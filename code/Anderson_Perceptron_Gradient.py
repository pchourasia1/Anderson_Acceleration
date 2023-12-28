import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn import metrics
from numpy import mean
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.special import softmax
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.model_selection import train_test_split

from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit

def softmax(z, epsilon=1e-10):
    # Subtract the maximum value from the logits
    z = z - np.max(z, keepdims=True)
    
    # Compute the exponentiated logits
    exp_z = np.exp(z)
    
    # Normalize the exponentiated logits to obtain the class probabilities
    return exp_z / (np.sum(exp_z, keepdims=True) + epsilon)

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
    
    
    check = pd.DataFrame(roc_auc_dict.items())
    return mean(check)
    
# X = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/spike/kmer_Frequency_Vector_7000_PCA_500.npy")
# attribute_data = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/spike/seq_data_variant_names_7000.npy")

# embeddings = "kmer/minimizer/spacedKmer"


# embeddings = "kmer"
# embeddings = "minimizer"
embeddings = "spacedKmer"

# X = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/spike/kmer_Frequency_Vector_7000_PCA_500.npy")
# X = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/spike/Minimizer_Frequency_Vector_7000_PCA_500.npy")
# X = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/spike/Spaced_kmers_7000_Freq_vec_g_9_k_4_PCA_500.npy")
# attribute_data = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/spike/seq_data_variant_names_7000.npy")
# print("Running the Embedding ", embeddings , "and Dataset Spike")

# X = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/host/Spike2Vec_on_unaligned_for_Host_Classification_Data_5558_seq_PCA_500.npy")
# X = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/host/Minimizer_on_unaligned_for_Host_Classification_Data_5558_seq_PCA_500.npy")
X = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/host/Spaced_kmers_5558_Freq_vec_k_4_PCA_500.npy")
attribute_data = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/host/attributes.npy")
print("Running the Embedding ", embeddings , "and Dataset host")



# X = np.load("/alina-data1/prakash/Anderson/data/embeddings/genome/Spike2Vec_on_unaligned_for_Host_Classification_Data_5558_seq_PCA_500.npy")
# X = np.load("/alina-data1/prakash/Anderson/data/embeddings/genome/Minimizer_on_unaligned_for_Host_Classification_Data_5558_seq_PCA_500.npy")
# X = np.load("/alina-data1/prakash/Anderson/data/embeddings/genome/Spaced_kmers_5558_Freq_vec_k_4_PCA_500.npy")
# attribute_data = np.load("/alina-data1/prakash/Anderson/data/embeddings/genome/all_attributes_8220.npy")
# print("Running the Embedding ", embeddings , "and Dataset Genome")


# X = np.load("/alina-data1/prakash/Anderson/data/embeddings/shortRead/kmer_frequency_vector_8140_seq.npy")
# X = np.load("/alina-data1/prakash/Anderson/data/embeddings/shortRead/minimizer_frequency_vector_8140_seq.npy")
# X = np.load("/alina-data1/prakash/Anderson/data/embeddings/shortRead/spaced_kmer_frequency_vector_8140_seq.npy")
# attribute_data = np.load("/alina-data1/prakash/Anderson/data/embeddings/shortRead/attribute_name_8140_seq.npy")
# print("Running the Embedding ", embeddings , "and Dataset short reads")


attr_new = []
for i in range(len(attribute_data)):
    aa = str(attribute_data[i]).replace("[","")
    aa_1 = aa.replace("]","")
    aa_2 = aa_1.replace("\'","")
    attr_new.append(aa_2)

unique_hst = list(np.unique(attr_new))

int_hosts = []
for ind_unique in range(len(attr_new)):
    variant_tmp = attr_new[ind_unique]
    ind_tmp = unique_hst.index(variant_tmp)
    int_hosts.append(ind_tmp)
    
print("Attribute data preprocessing Done")

y = np.array(int_hosts[:])

sss = ShuffleSplit(n_splits=1, test_size=0.3)

sss.get_n_splits(X, y)
train_index, test_index = next(sss.split(X, y)) 

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=68)



alpha_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
learning_rate_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, .1]
n_iterations = 3000

for learning_rate_val in learning_rate_values:

    losses = []
    metricsResults = []
    for alpha in alpha_values:
        print("Run Perceptron for learning rate ", learning_rate_val, " Alpha : ", alpha)
        #w, loss, acc = perceptron_aa_2(X_train, y_train, n_iterations, alpha)
        
        
        ##################################
        # Convert the labels to one-hot encoded labels
        y_one_hot = np.eye(np.max(y_train) + 1)[y_train]
        
        # Initialize the weight vector, loss list, and accuracy list
    #     w = np.random.randn(X_train.shape[1])
        w = np.random.randn(X_train.shape[1],X_train.shape[1])
    #     w = np.random.randn(n_features,n_features)
    #     w = np.random.randn(X_train.shape[1])
        
        loss = []
        accuracy = []
        
        # Initialize the weight vector history
        w_history = [w]
        
        # Run the Perceptron algorithm with Anderson acceleration
        for i in range(n_iterations):
            # Initialize the loss and the number of correct predictions for this iteration
            iter_loss = 0
            n_correct = 0
            
            # Initialize the gradient
            grad = np.zeros(X_train.shape[1])
            
            # Loop through the training data and compute the gradient
    #         for x, y in zip(X_train, y_one_hot):
            for x, y in zip(X_train, y_train):
                # Predict the class using the current weight vector
                y_pred = np.dot(w, x)
                
                # Normalize the prediction
                y_pred = y_pred / np.sum(y_pred)
                
                y_pred=softmax(y_pred)
                
                # Compute the gradient for this sample
                grad += y - y_pred
                
                # Compute the loss for this sample
                sample_loss = -np.sum(y * np.log(y_pred + learning_rate_val))
                
                # Increment the loss and the number of correct predictions for this iteration
                iter_loss += sample_loss
                n_correct += int(np.argmax(y_pred) == np.argmax(y))
            
            # Average the gradient over the number of samples
            grad /= len(X_train)
            
            # Add the updated weight vector to the history
            w_history.append(w)
            
            # If we have enough past weight vectors, perform Anderson acceleration
            if len(w_history) > 2:
                # Compute the difference between the current weight vector and the previous one
                diff = w_history[-1] - w_history[-2]
                
                # Update the current weight vector using Anderson acceleration
                w += alpha * diff + grad
            else:
                # Update the current weight vector using the gradient
                w += grad
            
            # Append the loss and the accuracy to their respective lists
            loss.append(iter_loss / len(X_train))
            accuracy.append(n_correct / len(X_train))
        
        
        ######################################
        
        
        losses.append(loss)
        # Make predictions on the test set
        y_pred = np.dot(X_test, w)
        
        # Compute the accuracy on the test set
        y_pred_all = []
        for ii in range(len(y_pred)):
            y_pred_all.append(np.argmax(y_pred[ii]))
        test_accuracy = np.mean(y_pred_all == y_test)
        print("For Alpha Value: " + str(alpha) + ", test accuracy: ",test_accuracy)
        
        accuracy = accuracy_score(y_test, y_pred_all)
        precision = precision_score(y_test, y_pred_all, average='macro')
        recall = recall_score(y_test, y_pred_all, average='macro')
        f1_weighted = metrics.f1_score(y_test, y_pred_all,average='weighted')
        f1_macro = metrics.f1_score(y_test, y_pred_all,average='macro')
        roc_auc = roc_auc_score_multiclass(y_test, y_pred_all, average='macro')[1]
        
        temp =[accuracy, precision, recall, f1_weighted, f1_macro, roc_auc]
        metricsResults.append(temp)
       


    for i, loss in enumerate(losses):
        plt.plot(loss, label=f'alpha={alpha_values[i]:.1f}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show() 
    plt.savefig("/alina-data1/prakash/Anderson/results/Perceptron/Host/"+ embeddings + "/Perceptron_LearningRate_"+ str(learning_rate_val) +"_Iterations_"+ str(n_iterations) +".png")   

    aa_loss_val = pd.DataFrame(losses).T
    aa_loss_val.to_csv("/alina-data1/prakash/Anderson/results/Perceptron/Host/"+ embeddings + "/Losses_Perceptron_LearningRate_"+ str(learning_rate_val) +"_Iterations_"+ str(n_iterations) +".csv")
    
    metrics_val = pd.DataFrame(metricsResults)
    metrics_val.to_csv("/alina-data1/prakash/Anderson/results/Perceptron/Host/"+ embeddings +"/Metric_Results_Perceptron_LearningRate_"+ str(learning_rate_val) +"_Iterations_"+ str(n_iterations) +".csv")
    
    
    