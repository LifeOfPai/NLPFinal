# Anjali Pai, 05/04/17, final Project- 
# Build a classifier that can identify a political leaning based on a sample of text

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
import textblob
from textblob import Blobber
from textblob.classifiers import MaxEntClassifier

def NaiveBayes(trainingVector):
    return nltk.NaiveBayesClassifier.train(trainingVector)

#    for t in testing:
#        prob = NB.prob_classify(t[0])
#        samples = prob.samples()
#        for s in samples:
#            print prob.prob(s), s

#    accuracy= nltk.classify.accuracy(NB, testing) * 100
#    NB.show_most_informative_features()
#    print accuracy

# Given a sentence, identifies if the sentence is progressive, not progressive, or neutral

def isProgressive(testset, trainset):
    prog_model = NaiveBayes(trainset)

    setProb = 1.0
    prob_list = []
    for t in testset:
        prob_list.append(prog_model.prob_classify(t))
        print prog_model.classify(t)
    print prob_list
    prog_model.show_most_informative_features()

#Parse text into feature vectors
# Column F (q1), G (confidence), H (q2), I (confidence)
# Column O (target), Column P (tweet)

tweets = {}
pos_tweets = []
neg_tweets = []
neut_tweets = []
def stage1Parse(path, n):
    stopWords = set(stopwords.words('english'))
    import csv
    import sys
    import copy
    from textblob import TextBlob
    csv.field_size_limit(sys.maxsize)
    sentences = []
    with open(path, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='\"',quoting= csv.QUOTE_ALL, skipinitialspace = True)
        for r in reader:
            sentences.append(r)
        
   # sentences = nltk.sentence_tokenize(text)
    import textblob
    from textblob.en.np_extractors import ConllExtractor
    tweet_i = sentences[0].index("tweet")
    target_i = sentences[0].index("target")
    sent_set = [s for s in sentences[1:n] if len(s)>15]
    all_words = {}
    tweet_words =[]
    blob_feats=[]
    grams = []
    chunks =[]
    ce = ConllExtractor()
    for s in sent_set:
        tweet = s[tweet_i].decode('ascii', errors="replace")
        nps = ce.extract(tweet)
        blob = TextBlob(tweet)
        ngrams= blob.ngrams(n=3)
        target = s[target_i]
        quest1 = s[5]
        conf_q1 = s[6]
        if conf_q1 > 0.5:
            words = [t for t in tweet.split() if t not in stopWords]
            tweet_words.append(tuple((words, quest1)))
            for w in words:
                all_words[w] = False

        if "AGAINST" in quest1:
            blob_feats.append(tuple((tweet, "neg")))
            grams.append(tuple(((" ".join(b) for b in ngrams), "neg")))
            for c in nps:
                chunks.append(tuple((c, "neg")))
        elif "FOR" in quest1:
            blob_feats.append(tuple((tweet, "pos")))
            grams.append(tuple(((" ".join(b) for b in ngrams), "pos")))
            for c in nps:
                chunks.append(tuple((c, "pos")))
        elif "NEUTRAL" in quest1:
            blob_feats.append(tuple((tweet, "neut")))
            grams.append(tuple(((" ".join(b) for b in ngrams), "neut")))
            for c in nps:
                chunks.append(tuple((c, "neut")))

    
    for tweet in tweet_words:
        feature = copy.deepcopy(all_words)
        for t in tweet[0]:
            feature[t] = True
        if "AGAINST:" in tweet[1]:
            sentiment = "negative"
            neg_tweets.append(tuple((feature,sentiment)))
        elif "FOR:" in tweet[1]:
            sentiment = "positive"
            pos_tweets.append(tuple((feature,sentiment)))
        elif "NEUTRAL:" in tweet[1]: 
            sentiment= "neutral"
            neut_tweets.append(tuple((feature,sentiment)))
   # print pos_tweets
    features = []
    features.extend(pos_tweets)
    features.extend(neg_tweets)
    features.extend(neut_tweets)
#        print features[0]
    return features

def progressiveModel(trainset):
    return NaiveBayes(trainset)

def stage2Parse(path):
    stopWords = set(stopwords.words('english'))
    import csv
    import sys
    import textblob
    csv.field_size_limit(sys.maxsize)
    sentences = []
    with open(path, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='\"',quoting= csv.QUOTE_ALL, skipinitialspace = True)
        for r in reader:
            sentences.append(r)

    tweet_i = sentences[0].index("text")
    bias_i = sentences[0].index("bias")
    
    all_words = {}
    text_words =[]
    for s in sentences[1:]:
        text = s[tweet_i]
        bias = s[bias_i]
        conf = s[bias_i +1]
        if conf > 0.5:
            words = [t for t in text.split() if t not in stopWords]
            text_words.append(tuple((words,bias)))
            for w in words:
                all_words[w] = False
            for txt in text_words:
                feature = all_words
                for t in txt[0]:
                    feature[t] = True
    # Add data specific feature vector creation


def getTweets(handle):
    import tweepy
    from tweepy import OAuthHandler
    import copy
    testset = []
    features={}
    consumer_key = "hKndCXwO0M48cPq6GTkpSLyci"
    consumer_secret = "JA0mTdQ5YLsbP3yhbmZJJMygLzmtxYsJL5hcumZNEoxYaSvrGM"
    access_key = "3347082569-vXtn1MmrtS0pZSxhP9PIB3nxoXvPi17GAIzwf5q"
    access_secret = "bczKyt7WJzv2rB8OxdDBebZ9HGvqcfLWpshJrCwNXOh3G"
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    twitter = tweepy.API(auth)
    source= twitter.user_timeline(screen_name=handle, count=100)
    tweets=[]
    for tweet in source:
        tweets.append(tweet.text)
        twt = tweet.text.split()
        for t in twt:
            features[t]=False
    for tweet in source:
        twt = tweet.text.split()
        feat = copy.deepcopy(features)
        for t in twt:
            feat[t]= True
        testset.append(features)
    
    return tweets
# Use this as a test set
def sentimentParse(path, start, end):
    stopWords = set(stopwords.words('english'))
    import csv
    import sys
    import copy
    from textblob import TextBlob
    csv.field_size_limit(sys.maxsize)
    sentences = []
    with open(path, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='\"',quoting= csv.QUOTE_ALL, skipinitialspace = True)
        for r in reader:
            sentences.append(r)
    
    tweet_i = sentences[0].index("tweet")

    features = []
    for s in sentences[start:end]:
        tweet = s[tweet_i].decode('ascii', errors="replace")
        sent = s[5]
        conf = s[6]
        blob = TextBlob(tweet)
        if conf > 0.5:
            ngrams = blob.ngrams(n=3)
            if "FOR" in sent:
                features.append(tuple((tweet, "Democrat")))
            elif "AGAINST" in sent:
                features.append(tuple((tweet, "Republican")))
    
    return features

def updateParse(path):
    f = open(path, "r")
    text = f.read()
    features = text.split(",")
    return features

def NounExtract(text):
    from textblob.en.np_extractors import ConllExtractor
    stopWords = set(stopwords.words('english')) 
    words =[w for w in text.split() if w not in stopWords]
    ce= ConllExtractor()
    np = ce.extract(text)
    positivity = VaderSentiment(text)
    if positivity>0:
        sent = "pos"
    else:
        sent = "neg"
    feats = {}
    for n in np:
        feats[(n,sent)]= True
    for w in words:
        feats[(w,sent)]=True
    return feats

def blobModel(train):
    import textblob
    from textblob.classifiers import NaiveBayesClassifier
    cl = NaiveBayesClassifier(train)
    return cl

def blobClassify(model,test):
    return model.classify(test)

def VaderSentiment(text):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    VS = SentimentIntensityAnalyzer()
    return VS.polarity_scores(text)['compound']

def VaderFeatures(text):
    compound = VaderSentiment(text)
    
    
def maxEntropyModel(trainset):
    mec = MaxEntClassifier(trainset, feature_extractor=NounExtract) #insert parameters
    return mec

def maxEntropyModelNLTK(trainset):
    from nltk.classify import maxent
    mec = maxent().train(trainset)
    return mec

def DemocraticModel():
    import textblob
    from textblob import TextBlob
    from textblob.classifiers import MaxEntClassifier

    tweets = getTweets("TheDemocrats")
    features = []
    for t in tweets:
#        blob = TextBlob(t)
#        ngrams = blob.ngrams(n=3)
        features.append(tuple((t, "Democrat")))
       # for n in ngrams:
       #     features.append(tuple((n, "Democrat")))

    repTweets = getTweets("SenateGOP")

    for t in repTweets:
#        blob = TextBlob(t)
#        ngrams = blob.ngrams(n=3)
        features.append(tuple((t, "Republican")))
   #     for n in ngrams:
   #        features.append(tuple((n, "Republican")))
    # sentiments = sentimentParse("progressive-tweet-sentiment.csv")
   # features.extend(sentiments)

    model = maxEntropyModel(features)

    return model

missed= []
def RepClassifications(testset, model):
    TP = 0
    FP = 0
    FN = 0

    for t in testset:
        if t[1]== "Republican":
            if model.classify(t[0]) =="Republican":
                TP +=1
            else:
                FN +=1
                missed.append(t)
              #  print(t),","
        elif model.classify(t[0])=="Republican" and t[1]=="Democrat":
            FP +=1
            missed.append(t)
           # print(t),","    
    return tuple((TP, FP,FN))

missed = []
def DemClassifications(testset, model):
    TP = 0
    FP = 0
    FN = 0

    for t in testset:
        if t[1]== "Democrat":
            if model.classify(t[0]) =="Democrat":
                TP +=1
            else:
                FN +=1
        elif model.classify(t[0])=="Democrat" and t[1]=="Republican":
            FP +=1
    
    return tuple((TP, FP,FN))

def F1(cl, model):
   
    TP = cl[0]
    FP = cl[1]
    FN= cl[2]
   
    if TP == 0:
        F1 = 0
    else:
        prec = float(TP)/float((TP+FP))
        rec = float(TP)/float((TP+FN))
        F1 = (2*prec*rec)/(prec+rec)
    return F1


def main():
    RD = DemocraticModel()
  
    testset = sentimentParse("progressive-tweet-sentiment.csv",600,700)
    RepClassifications(testset, RD)
  
    RD.update(missed)
   
    set2 = getTweets("SpeakerRyan")
    testset2 = [(t, "Republican") for t in set2]
    set2_dem = getTweets("NancyPelosi")
    testset2.extend([(t, "Democrat") for t in set2_dem])
    rep_class2 = RepClassifications(testset, RD)
    dem_class2 = DemClassifications(testset2, RD)
    print RD.accuracy(testset2), F1(rep_class2, RD), F1(dem_class2, RD)
   
    testset3= sentimentParse("progressive-tweet-sentiment.csv",400,500)
    rep_class3 = RepClassifications(testset3, RD)
    dem_class3 = DemClassifications(testset3, RD)
    print RD.accuracy(testset3), F1(rep_class3, RD), F1(dem_class3, RD)
   

main()
