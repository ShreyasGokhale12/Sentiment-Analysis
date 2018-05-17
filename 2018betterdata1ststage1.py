import nltk
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self , *classifiers):  #passing a list of classifiers
        self._classifiers = classifiers  #for all classifiers

    def classify(self , features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        return mode(votes)

    def confidence(self , features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


short_pos = open("positive.txt","r", encoding='utf-8', errors='replace').read()
short_neg = open("negative.txt","r", encoding='utf-8', errors='replace').read()

documents = []
all_words = []
allowed_word_types = ['J']

for r in short_pos.split('\n'):
    documents.append(('r','pos'))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for r in short_neg.split('\n'):
    documents.append(('r','neg'))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
            
#---------------------------------------------------------------------------------------------------------------------#        

saved_documents = open('pickled_algos/documents.pickle','wb')
pickle.dump(documents , saved_documents)
saved_documents.close()

#---------------------------------------------------------------------------------------------------------------------#            

all_words = nltk.FreqDist(all_words)
word_features = list(all_words)[:5000]

saved_word_features = open('pickled_algos/word_features.pickle','wb')
pickle.dump(word_features , saved_word_features)
saved_word_features.close()

#---------------------------------------------------------------------------------------------------------------------#

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words) #returns boolean whether that word is present in top 5000 words
        
    return features

featuresets = [(find_features(sentence),category) for (sentence , category) in documents]

saved_featureset = open('pickled_algos/featuresets.pickle','wb')
pickle.dump(featuresets ,saved_featureset)
saved_featureset.close()

#---------------------------------------------------------------------------------------------------------------------#

random.shuffle(featuresets)

print(len(featuresets))

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

#---------------------------------------------------------------------------------------------------------------------#

classifier = nltk.NaiveBayesClassifier.train(training_set)
print('Original Accuracy on the training set in percent : ' , nltk.classify.accuracy(classifier , testing_set)*100)
classifier.show_most_informative_features(15)

save_classifier = open('pickled_algos/NaiveBayesClassifier','wb')
pickle.dump(classifier , save_classifier)
save_classifier.close()

#---------------------------------------------------------------------------------------------------------------------#

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open('pickled_algos/MNB_classifier','wb')
pickle.dump(MNB_classifier , save_classifier)
save_classifier.close()

#---------------------------------------------------------------------------------------------------------------------#

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open('pickled_algos/BernoulliNB_classifier','wb')
pickle.dump(BernoulliNB_classifier , save_classifier)
save_classifier.close()

#---------------------------------------------------------------------------------------------------------------------#

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open('pickled_algos/LogisticRegression_classifier','wb')
pickle.dump(LogisticRegression_classifier , save_classifier)
save_classifier.close()

#---------------------------------------------------------------------------------------------------------------------#

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

save_classifier = open('pickled_algos/SGDClassifier_classifier','wb')
pickle.dump(SGDClassifier_classifier , save_classifier)
save_classifier.close()

#---------------------------------------------------------------------------------------------------------------------#

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open('pickled_algos/LinearSVC_classifier','wb')
pickle.dump(LinearSVC_classifier , save_classifier)
save_classifier.close()

#---------------------------------------------------------------------------------------------------------------------#
voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier)




def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
                               
