# Live Sentiment Analysis of Twitter

6 classifiers - NaiveBayesClassifier , MultinomialNB , BernoulliNB , LogisticRegression , SGDClassifier , LinearSVC are used .The dataset consists of two text files - one which contains positive MOVIE reviews (positive.txt) and other negative MOVIE reviews (negative.txt) . All these classifiers were passed in class named 'VoteClassifier' which returns the confidence value .

This model is saved as sentiment_mod.py

sentiment_mod.py is loaded into the file twittersentimentanalysis.py The keys are obtained by creating a Twitter App from Twitter account. By using the module 'tweepy' and json , all live tweets on a particular topic say 'car' are loaded . These tweets are then passed through the above model . The tweets having confidence value greater than or equal to 0.8 are saved in a file 'twitter-out.txt'

Live Matlotlib plotting is then used to create a graph of the tweets saved in 'twitter-out.txt' 




