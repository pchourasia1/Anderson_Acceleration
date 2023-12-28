from sklearn.svm import LinearSVC
from sklearn import metrics
from numpy import mean
from sklearn.metrics import roc_auc_score


class LinearSVCWithAA(LinearSVC):
    def __init__(self, learning_rate=1.0, alpha=0.1, n_iterations=10, verbose=False):        
        # Initialize the attributes of the LinearSVCWithAA class
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.loss = []  # Add this line to store the loss values

#         # Initialize the coef_ attribute
# #         self.coef_ = None
#         self.coef_ = np.random.randn(5,)
    
#         # Initialize the coef_prev attribute
#         self.coef_prev = self.coef_
      
    
    
    def fit(self, X, y, n_iterations=100, alpha=0.1,learning_rate=0.1):
        """
        Fit the model to the training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target labels.
        n_iterations : int
            Number of iterations to run the algorithm.
        alpha : float
            Anderson acceleration factor.
        """
        # Convert the labels to one-hot encoded labels
        y_one_hot = np.eye(np.max(y) + 1)[y]

        # Initialize the weight vector and the loss list
#         self.coef_ = np.random.randn(X.shape[1])
#         self.coef_ = np.random.randn(X.shape[1], len(np.unique(y)))
        self.coef_ = np.random.randn(X.shape[1], X.shape[1])
#         loss = []

        # Initialize the weight vector history
        coef_history = [self.coef_]

        # Run the Perceptron algorithm with Anderson acceleration
        for i in range(n_iterations):
            # Compute the gradient
            grad = self._gradient(X, y)

            # Add the updated weight vector to the history
            coef_history.append(self.coef_)

            # If we have enough past weight vectors, perform Anderson acceleration
            if len(coef_history) > 2:
                # Compute the difference between the current weight vector and the previous one
                diff = coef_history[-1] - coef_history[-2]

                # Update the current weight vector using Anderson acceleration
#                 print("self.coef_ ",self.coef_)
#                 print("self.coef_ ",self.coef_)
                self.coef_ += alpha * diff + grad[0]
            else:
                # Update the current weight vector using the gradient
#                 print("self.coef_ ",self.coef_)
#                 print("grad ",grad[0])
                self.coef_ += learning_rate * grad[0]

            # Compute the loss for this iteration
            y_pred = self.predict(X)
#             iter_loss = self.hinge_loss(y,y_pred)
#             iter_loss = np.maximum(0, 1 - y * y_pred)
#             print("iter_loss: ",iter_loss)
            #######################################################
            # Compute the loss for this sample
#             sample_loss = -np.sum(y * np.log(y_pred))
            # Increment the loss and the number of correct predictions for this iteration
#             iter_loss += int(sample_loss)
#             n_correct += int(np.argmax(y_pred) == np.argmax(y))
            # Compute the loss for this iteration
#             loss_value = np.maximum(0, 1 - y * y_pred) # hinge loss
            loss_value = self.cross_entropy_loss(y, y_pred)
#             print("Iteration: ", i, ", Loss: ",loss_value)
            self.loss.append(loss_value)
#             self.loss.append(iter_loss)
            #########################################################

            # Save the previous weight vector
            self.coef_prev = self.coef_

        # Save the number of iterations and the alpha value
        self.n_iterations_ = n_iterations
        self.alpha_ = alpha

    def cross_entropy_loss(self, y, y_pred):
        # Compute the loss for each sample
#         print("y length: ",len(y))
#         print("y_pred length: ",len(y_pred))
        sample_loss = -np.sum(y * np.log(y_pred + 1e-10))

        # Average the loss over the number of samples
        loss = np.mean(sample_loss)
#         print("sample_loss",sample_loss,", loss",loss)

        return loss


    def hinge_loss(self, X, y):
        # Compute the prediction for each sample
        y_pred = self.predict(X)
        
        # Compute the hinge loss for each sample
        loss = np.maximum(0, 1 - y * y_pred)
        
        return loss
    
    
    def predict(self, X):
#         # Compute the class scores for each sample
#         scores = np.dot(X, self.coef_.T)

#         # Predict the class for each sample
#         y_pred = np.argmax(scores)

#         return y_pred
    
        # Predict the class of each sample in X
        y_pred = np.dot(X, self.coef_)

        # Normalize the predictions
        y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)

        # Return the predicted class for each sample
        return np.argmax(y_pred, axis=1)


    
    def _gradient(self, X, y):
        # Compute the prediction for each sample
        y_pred = self.predict(X)

        # Compute the gradient of the loss function with respect to the weight vector
        
        grad_coef = np.dot(X.T, np.maximum(0, 1 - y * y_pred))
        
        # Average the gradient over the number of samples
#         grad_coef /= len(X)
        grad_coef = np.divide(grad_coef, len(X))
        
        grad_coef = np.squeeze(grad_coef) ###################################################

        # Compute the gradient of the loss function with respect to the bias term
        grad_intercept = np.sum(np.maximum(0, 1 - y * y_pred)) / len(X)

        # Return the gradient as a tuple
        return (grad_coef, grad_intercept)
    
   
    
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

# embeddings = "kmer"
# embeddings = "minimizer"
embeddings = "spacedKmer"
# embeddings = "kmer/minimizer/spacedKmer"

# X = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/spike/kmer_Frequency_Vector_7000_PCA_500.npy")
# X = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/spike/Minimizer_Frequency_Vector_7000_PCA_500.npy")
# X = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/spike/Spaced_kmers_7000_Freq_vec_g_9_k_4_PCA_500.npy")
# attribute_data = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/spike/seq_data_variant_names_7000.npy")
# print("Running the Embedding ", embeddings , "and Dataset Spike")

# X = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/host/Spike2Vec_on_unaligned_for_Host_Classification_Data_5558_seq_PCA_500.npy")
# X = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/host/Minimizer_on_unaligned_for_Host_Classification_Data_5558_seq_PCA_500.npy")
# X = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/host/Spaced_kmers_5558_Freq_vec_k_4_PCA_500.npy")
# attribute_data = np.load("/alina-data1/prakash/Anderson/data/pca_embeddings/host/attributes.npy")
# print("Running the Embedding ", embeddings , "and Dataset host")


# X = np.load("/alina-data1/prakash/Anderson/data/embeddings/genome/Spike2Vec_on_unaligned_for_Host_Classification_Data_5558_seq_PCA_500.npy")
# X = np.load("/alina-data1/prakash/Anderson/data/embeddings/genome/Minimizer_on_unaligned_for_Host_Classification_Data_5558_seq_PCA_500.npy")
# X = np.load("/alina-data1/prakash/Anderson/data/embeddings/genome/Spaced_kmers_5558_Freq_vec_k_4_PCA_500.npy")
# attribute_data = np.load("/alina-data1/prakash/Anderson/data/embeddings/genome/all_attributes_8220.npy")
# print("Running the Embedding ", embeddings , "and Dataset Genome")


# X = np.load("/alina-data1/prakash/Anderson/data/embeddings/shortRead/kmer_frequency_vector_8140_seq.npy")
# X = np.load("/alina-data1/prakash/Anderson/data/embeddings/shortRead/minimizer_frequency_vector_8140_seq.npy")
X = np.load("/alina-data1/prakash/Anderson/data/embeddings/shortRead/spaced_kmer_frequency_vector_8140_seq.npy")
attribute_data = np.load("/alina-data1/prakash/Anderson/data/embeddings/shortRead/attribute_name_8140_seq.npy")
print("Running the Embedding ", embeddings , "and Dataset short reads")

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

# X = X[0:50]
# y = y[0:50]
sss = ShuffleSplit(n_splits=1, test_size=0.3)

sss.get_n_splits(X, y)
train_index, test_index = next(sss.split(X, y)) 

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=68)


# Train the model using the LinearSVCWithAA class
# model = LinearSVCWithAA()
# model.fit(X_train, y_train,n_iterations=10, learning_rate=0.1)

# Predict the classes of the test set
# y_pred = []
# for i in range(len(X_test)):
#     y_pred.append(model.predict(X_test[i]))


# Create a LinearSVCWithAA object with Anderson acceleration

alpha_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
learning_rate_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, .1]
n_iterations_value = 3000

# Create a LinearSVCWithAA object without Anderson acceleration
# svc_no_aa = LinearSVCWithAA(alpha=0, learning_rate=learning_rate_val)
# Fit the models on the training data
# svc_no_aa.fit(X_train, y_train,n_iterations=n_iterations_value, learning_rate=learning_rate_val)

for learning_rate_val in learning_rate_values:

    plt.figure()
    losses = []
    metricsResults = []

    for alpha in alpha_values:
        print("Running for Alpha = ", alpha, "Learning Rate = ", learning_rate_val)
        svc_aa = LinearSVCWithAA(alpha=alpha, learning_rate=learning_rate_val)
        svc_aa.fit(X_train, y_train,n_iterations=n_iterations_value, learning_rate=learning_rate_val)
        losses.append(svc_aa.loss)
        # Plot the loss for the two models
        y_pred = svc_aa.predict(X_test)
        # Compute the classification accuracy
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
        f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
        roc_auc = roc_auc_score_multiclass(y_test, y_pred, average='macro')[1]
        
        temp =[accuracy, precision, recall, f1_weighted, f1_macro, roc_auc]
        metricsResults.append(temp)
        
        plt.plot(svc_aa.loss, label="alpha: " + str(alpha))
       # plt.plot(svc_no_aa.loss, label='Without Anderson acceleration')
       
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("/alina-data1/prakash/Anderson/results/SVM/ShortRead/"+embeddings+"/SVM_LearningRate_"+ str(learning_rate_val) +"_Iterations_"+ str(n_iterations_value) +".png")
    aa_loss_val = pd.DataFrame(losses).T
    aa_loss_val.to_csv("/alina-data1/prakash/Anderson/results/SVM/ShortRead/"+embeddings+"/Losses_SVM_LearningRate_"+ str(learning_rate_val) +"_Iterations_"+ str(n_iterations_value) +".csv")

    metrics_val = pd.DataFrame(metricsResults)
    metrics_val.to_csv("/alina-data1/prakash/Anderson/results/SVM/ShortRead/"+embeddings+"/Metric_Results_SVM_LearningRate_"+ str(learning_rate_val) +"_Iterations_"+ str(n_iterations_value) +".csv")
    
    


