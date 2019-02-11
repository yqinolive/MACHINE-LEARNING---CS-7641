#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 11:23:09 2019

@author: OliveQIn
"""
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
#nltk.download() 
from gensim.models import word2vec, KeyedVectors
import numpy as np


train_mv = pd.read_csv( "labeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )

df_alexa = pd.read_csv( "amazon_alexa.tsv", header=0, delimiter="\t", quoting=3 )


# Train Word2Vec
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )


def review_to_wordlist( review, remove_stopwords=False ):
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)
    
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    return sentences


sentences = []  # Initialize an empty list of sentences

print("Parsing sentences from training set")
for review in train_mv["review"]:
    sentences += review_to_sentences(review, tokenizer)


print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

    
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words


model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

model.init_sims(replace=True)


model.wv.save_word2vec_format('300features_40minwords_10context.bin')

model.wv.save_word2vec_format('300features_40minwords_10context.txt', binary=False)

model_name = "300features_40minwords_10context"
model.save(model_name)


model.wv.save_word2vec_format('300features_40minwords_10context.txt', binary=False)


'''
# call saved model
model = KeyedVectors.load_word2vec_format("300features_40minwords_10context.bin")
'''


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs


clean_train_reviews = []
for review in train_mv["review"]:
    clean_train_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )


trainDataVecs=np.nan_to_num(trainDataVecs)


clean_train_reviews = []
for review in df_alexa['verified_reviews']: 
    clean_train_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
trainDataVecs_alexa = getAvgFeatureVecs( clean_train_reviews, model, num_features )

trainDataVecs_alexa=np.nan_to_num(trainDataVecs_alexa)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import *



X_train_mv = trainDataVecs[0:20000,] 
y_train_mv = train_mv['sentiment'][0:20000]

X_test = trainDataVecs[20000:25000,] 
y_test = train_mv['sentiment'][20000:25000]




X_train_alexa = trainDataVecs_alexa[0:2500,] 
y_train_alexa = df_alexa['feedback'][0:2500]

X_test_alexa = trainDataVecs_alexa[2500:,] 
y_test_alexa = df_alexa['feedback'][2500:]




def DTClassfier(X_train_mv, X_test, y_train_mv, y_test,ndepth,nn,pprint=False):
    clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = ndepth, random_state=1)
    clf.fit(X_train_mv, y_train_mv)
    
    y_pred = clf.predict(X_train_mv)
    y_pred_test = clf.predict(X_test)
    
    if pprint:
        print(nn)
        print('Decision Tree: ')
        print('training data metrics: ')
        print('accuracy score: ', accuracy_score(y_train_mv, y_pred)*100)
        print('precision: ', precision_score(y_train_mv, y_pred)*100)
        print('recall: ', recall_score(y_train_mv, y_pred)*100)
        print('F1_score: ', f1_score(y_train_mv, y_pred)*100)    
        #print('confusion matrix: ',   confusion_matrix(y_train_mv, y_pred)) 
        print('testing data metrics: ')
        print('accuracy score: ', accuracy_score(y_test, y_pred_test)*100)
        print('precision: ', precision_score(y_test, y_pred_test)*100)
        print('recall: ', recall_score(y_test, y_pred_test)*100)
        print('F1_score: ', f1_score(y_test, y_pred_test)*100)
    return [accuracy_score(y_train_mv, y_pred)*100, accuracy_score(y_test, y_pred_test)*100]




DT_res_mv = []
for i in range(2,10,2):
    tt = DTClassfier(X_train_mv, X_test, y_train_mv, y_test,i,'Movie Review')
    tt.append(i)
    DT_res_mv.append(tt)
DT_res_mv = pd.DataFrame(DT_res_mv)


plt.plot(DT_res_mv.iloc[:,2], DT_res_mv.iloc[:,0],label="Train_accuracy")
plt.plot(DT_res_mv.iloc[:,2], DT_res_mv.iloc[:,1],label="Test_accuracy")
plt.title('Movie Decision Tree Accuracy Analysis')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.legend()

DTClassfier(X_train_mv, X_test, y_train_mv, y_test,6,'Movie Review',pprint=True)


DT_res_Alexa = []
for i in range(2,10,2):
    tt = DTClassfier(X_train_alexa, X_test_alexa, y_train_alexa, y_test_alexa,i,'Alexa Review')
    tt.append(i)
    DT_res_Alexa.append(tt)
DT_res_Alexa = pd.DataFrame(DT_res_Alexa)


plt.plot(DT_res_Alexa.iloc[:,2], DT_res_Alexa.iloc[:,0],label="Train_accuracy")
plt.plot(DT_res_Alexa.iloc[:,2], DT_res_Alexa.iloc[:,1],label="Test_accuracy")
plt.title('Alexa Review Decision Tree Accuracy Analysis')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.legend()


DTClassfier(X_train_alexa, X_test_alexa, y_train_alexa, y_test_alexa,3,'Alexa Review',pprint=True)


'''
Alexa Review
Decision Tree: 
training data metrics: 
accuracy score:  91.67999999999999
precision:  91.80064308681672
recall:  99.82517482517483
F1_score:  95.64489112227807
testing data metrics: 
accuracy score:  91.53846153846153
precision:  92.96875
recall:  98.34710743801654
'''


'''
from sklearn import tree
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render('tube') 

X_train_mv_df=pd.DataFrame(X_train_mv)

dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=X_train_mv_df.columns,  
                     class_names=y_train_mv.name,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data) 

'''







def NNClassifier(X_train_mv, X_test, y_train_mv, y_test,nn):    
    clf = MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(5,), random_state=1)
    clf.fit(X_train_mv, y_train_mv)
    y_pred = clf.predict(X_train_mv)
    
    print('Neural Net: ')
    print(nn)
    print('training data metrics: ')
    print('accuracy score: ', accuracy_score(y_train_mv, y_pred)*100)
    print('precision: ', precision_score(y_train_mv, y_pred)*100)
    print('recall: ', recall_score(y_train_mv, y_pred)*100)
    print('F1_score: ', f1_score(y_train_mv, y_pred)*100)    
    #print('confusion matrix: ',   confusion_matrix(y_train_mv, y_pred)) 
    y_pred_test = clf.predict(X_test)
    print('testing data metrics: ')
    print('accuracy score: ', accuracy_score(y_test, y_pred_test)*100)
    print('precision: ', precision_score(y_test, y_pred_test)*100)
    print('recall: ', recall_score(y_test, y_pred_test)*100)
    print('F1_score: ', f1_score(y_test, y_pred_test)*100)
    
    #return [accuracy_score(y_train_mv, y_pred)*100,accuracy_score(y_test, y_pred_test)*100]

NNClassifier(X_train_mv, X_test, y_train_mv, y_test,'movie')
NNClassifier(X_train_alexa, X_test_alexa, y_train_alexa, y_test_alexa,'alexa')


'''
(5,)
training data metrics: 
accuracy score:  86.005
precision:  85.57682769566512
recall:  86.51223425591657
F1_score:  86.04198872986586
testing data metrics: 
accuracy score:  84.76
precision:  85.03968253968253
recall:  84.77056962025317
F1_score:  84.90491283676704


(10,)
Neural Net: 
training data metrics: 
accuracy score:  86.5
precision:  85.9359557224748
recall:  87.19414360208584
F1_score:  86.56047784967646
testing data metrics: 
accuracy score:  85.08
precision:  85.13406940063092
recall:  85.40348101265823
F1_score:  85.26856240126382


(15,)
Neural Net: 
training data metrics: 
accuracy score:  87.03999999999999
precision:  86.97394789579158
recall:  87.0437224227838
F1_score:  87.00882117080994
testing data metrics: 
accuracy score:  85.32
precision:  85.70859872611464
recall:  85.16613924050634
F1_score:  85.43650793650795


(5,5)
Neural Net: 
training data metrics: 
accuracy score:  86.44500000000001
precision:  86.20001994216771
recall:  86.69273967107902
F1_score:  86.4456777161142
testing data metrics: 
accuracy score:  84.98
precision:  85.44076585560431
recall:  84.73101265822784
F1_score:  85.08440913604768


(5,5,5)
Neural Net: 
training data metrics: 
accuracy score:  86.77
precision:  86.39705882352942
recall:  87.19414360208584
F1_score:  86.79377121181872
testing data metrics: 
accuracy score:  85.0
precision:  85.22187004754358
recall:  85.0870253164557
F1_score:  85.1543942992874


5,10
Neural Net: 
training data metrics: 
accuracy score:  86.195
precision:  85.64508156203658
recall:  86.87324508624148
F1_score:  86.25479165629511
testing data metrics: 
accuracy score:  84.86
precision:  84.821077467558
recall:  85.3243670886076
F1_score:  85.07197791362651


5,10,10
Neural Net: 
training data metrics: 
accuracy score:  86.28
precision:  85.16248297334111
recall:  87.77577216205376
F1_score:  86.44938271604939
testing data metrics: 
accuracy score:  84.88
precision:  84.26140757927301
recall:  86.19462025316456
F1_score:  85.2170512319124
'''





def Boosting(X_train_mv, X_test, y_train_mv, y_test,nn):
    clf = GradientBoostingClassifier(max_depth=3,random_state=1)
    clf.fit(X_train_mv, y_train_mv)
    y_pred = clf.predict(X_train_mv)
    
    print('training data metrics: ')
    print('GBM')
    print(nn)
    print('accuracy score: ', accuracy_score(y_train_mv, y_pred)*100)
    print('precision: ', precision_score(y_train_mv, y_pred)*100)
    print('recall: ', recall_score(y_train_mv, y_pred)*100)
    print('F1_score: ', f1_score(y_train_mv, y_pred)*100)    
    #print('confusion matrix: ',   confusion_matrix(y_train_mv, y_pred)) 
    y_pred_test = clf.predict(X_test)
    print('testing data metrics: ')
    print('accuracy score: ', accuracy_score(y_test, y_pred_test)*100)
    print('precision: ', precision_score(y_test, y_pred_test)*100)
    print('recall: ', recall_score(y_test, y_pred_test)*100)
    print('F1_score: ', f1_score(y_test, y_pred_test)*100) 
    
    return clf
    
clf_mv=Boosting(X_train_mv, X_test, y_train_mv, y_test,'movie')

clf_alexa=Boosting(X_train_alexa, X_test_alexa, y_train_alexa, y_test_alexa,'alexa')


'''
Boosting
training data metrics: 
accuracy score:  85.42999999999999
precision:  84.85087892553823
recall:  86.16125150421179
F1_score:  85.50104488008758
testing data metrics: 
accuracy score:  83.14
precision:  83.13016122689737
recall:  83.62341772151899
F1_score:  83.37605994872807
'''

'''
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, scoring=None, obj_line=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    from sklearn.model_selection import learning_curve
    import numpy as np
    from matplotlib import pyplot as plt
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    if obj_line:
        plt.axhline(y=obj_line, color='blue')

    plt.legend(loc="best")
    return plt



plot_learning_curve(clf_mv,"Learning Curve -- Boosting",X_train_mv,y_train_mv,ylim=None, cv=None, scoring=None, obj_line=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5))


plot_learning_curve(clf_alexa,"Learning Curve -- Boosting",X_train_alexa,y_train_alexa,ylim=None, cv=None, scoring=None, obj_line=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5))

'''


# SVM training start here
def SVM_Kernal(X_train_mv, X_test, y_train_mv, y_test,nn, kernel = 'linear',random_state=1):
    clf = SVC(gamma='auto', kernel= kernel)
    clf.fit(X_train_mv, y_train_mv) 
    y_pred = clf.predict(X_train_mv)
    print('SVM: ')
    print(nn)
    print (kernel)
    print('training data metrics: ')
    print('accuracy score: ', accuracy_score(y_train_mv, y_pred)*100)
    print('precision: ', precision_score(y_train_mv, y_pred)*100)
    print('recall: ', recall_score(y_train_mv, y_pred)*100)
    print('F1_score: ', f1_score(y_train_mv, y_pred)*100)    
    #print('confusion matrix: ',   confusion_matrix(y_train_mv, y_pred)) 
    y_pred_test = clf.predict(X_test)
    print('testing data metrics: ')
    print('accuracy score: ', accuracy_score(y_test, y_pred_test)*100)
    print('precision: ', precision_score(y_test, y_pred_test)*100)
    print('recall: ', recall_score(y_test, y_pred_test)*100)
    print('F1_score: ', f1_score(y_test, y_pred_test)*100)    
    
SVM_Kernal(X_train_mv, X_test, y_train_mv, y_test, 'movie', kernel = 'linear')

SVM_Kernal(X_train_alexa, X_test_alexa, y_train_alexa, y_test_alexa, 'alexa', kernel = 'linear')


'''
'linear'
SVM: 
training data metrics: 
accuracy score:  85.015
precision:  84.26171529619806
recall:  86.01083032490975
F1_score:  85.12728896828943
testing data metrics: 
accuracy score:  84.11999999999999
precision:  84.05341712490181
recall:  84.65189873417721
F1_score:  84.3515963736697



SVM: 
rbf
training data metrics: 
accuracy score:  71.875
precision:  75.17664774701726
recall:  65.0822302446851
F1_score:  69.76619188390217
testing data metrics: 
accuracy score:  70.94
precision:  75.15208235844642
recall:  63.52848101265823
F1_score:  68.85316184351554


poly
training data metrics: 
accuracy score:  50.13999999999999
precision:  0.0
recall:  0.0
F1_score:  0.0
testing data metrics: 
accuracy score:  49.44
precision:  0.0
recall:  0.0
F1_score:  0.0


SVM: 
sigmoid
training data metrics: 
accuracy score:  50.78
precision:  90.50632911392405
recall:  1.4340152426795028
F1_score:  2.8232971372161897
testing data metrics: 
accuracy score:  50.1
precision:  83.6734693877551
recall:  1.6218354430379747
F1_score:  3.181994567326348




'''






def KNN(X_train_mv, X_test, y_train_mv, y_test,n,nn):
    clf = KNeighborsClassifier(n_neighbors=n,weights='distance')
    clf.fit(X_train_mv, y_train_mv) 
    y_pred = clf.predict(X_train_mv)
    print('KNN: ')
    print(nn)
    print('training data metrics: ')
    print('accuracy score: ', accuracy_score(y_train_mv, y_pred)*100)
    print('precision: ', precision_score(y_train_mv, y_pred)*100)
    print('recall: ', recall_score(y_train_mv, y_pred)*100)
    print('F1_score: ', f1_score(y_train_mv, y_pred)*100)    
    #print('confusion matrix: ',   confusion_matrix(y_train_mv, y_pred)) 
    y_pred_test = clf.predict(X_test)
    print('testing data metrics: ')
    print('accuracy score: ', accuracy_score(y_test, y_pred_test)*100)
    print('precision: ', precision_score(y_test, y_pred_test)*100)
    print('recall: ', recall_score(y_test, y_pred_test)*100)
    print('F1_score: ', f1_score(y_test, y_pred_test)*100)  
    return [accuracy_score(y_train_mv, y_pred)*100, accuracy_score(y_test, y_pred_test)*100]
    
KNN(X_train_mv, X_test, y_train_mv, y_test,3,'movie')
KNN(X_train_alexa, X_test_alexa, y_train_alexa, y_test_alexa,3,'alexa')


'''
cc=[]
for i in range(3,13,2):
    rr=KNN(X_train_mv, X_test, y_train_mv, y_test,i)
    rr.append(i)
    cc.append(rr)
cc=pd.DataFrame(cc)




plt.plot(cc.iloc[:,2], cc.iloc[:,0],label="Train_accuracy")
plt.plot(cc.iloc[:,2], cc.iloc[:,1],label="Test_accuracy")
plt.title('Movie Review KNN Accuracy Analysis')
plt.xlabel('Number of Nearest Neighbour')
plt.ylabel('Accuracy')
plt.legend()

'''


'''
KNN: 
training data metrics: 
accuracy score:  86.81
precision:  87.99212598425197
recall:  85.16847172081829
F1_score:  86.55727680391358
testing data metrics: 
accuracy score:  80.08
precision:  82.56802721088435
recall:  76.81962025316456
F1_score:  79.59016393442624
'''


















