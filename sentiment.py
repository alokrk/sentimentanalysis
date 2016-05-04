import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec

nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)

    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    #print train_pos[1]
    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # Determine a list of words that will be used as features.
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.

    posTrain=[]
    for words in train_pos:
        train_pos_vector = set(words) - stopwords
        posTrain.append(list(train_pos_vector))

    posThreshold = int(0.01*len(posTrain))

    #posTest=[]
    #for words in test_pos:
    #    test_pos_vector = set(words) - stopwords
    #    posTest.append(list(test_pos_vector))

    #posThreshold2 = int(0.01*len(posTest))

    count_posTrain = collections.Counter()
    for i in posTrain:
        for word in i:
            count_posTrain[word]+=1

    #count_posTest = collections.Counter()
    #for i in posTest:
    #    for word in i:
    #        count_posTest[word]+=1

    negTrain=[]
    for words in train_neg:
        train_neg_vector = set(words) - stopwords
        negTrain.append(list(train_neg_vector))

    negThreshold = int(0.01*len(negTrain))

    #negTest=[]
    #for words in test_neg:
    #    test_neg_vector = set(words) - stopwords
    #    negTest.append(list(test_neg_vector))

    #negThreshold2 = int(0.01*len(negTest))

    count_negTrain = collections.Counter()
    for i in negTrain:
        for word in i:
            count_negTrain[word]+=1

    #count_negTest = collections.Counter()
    #for i in negTrain:
    #    for word in i:
    #        count_negTest[word]+=1


    for i in list(count_posTrain):
        if count_posTrain[i] < posThreshold:
            del count_posTrain[i]

    for i in list(count_negTrain):
        if count_negTrain[i] < negThreshold:
            del count_negTrain[i]

    #for i in list(count_posTest):
    #    if count_posTest[i] < posThreshold2:
    #        del count_posTest[i]

    #for i in list(count_negTest):
    #    if count_negTest[i] < negThreshold2:
    #        del count_negTest[i]

    tempPos = count_posTrain
    tempNeg = count_negTrain

    for i in list(tempPos):

        if (tempPos[i] < 2*tempNeg[i] and tempNeg[i] < 2*tempPos[i]):
            del count_posTrain[i]
            del count_negTrain[i]

        if tempPos[i] < 2*tempNeg[i]:
            del count_posTrain[i]

        if tempNeg[i] < 2*tempPos[i]:
            del count_negTrain[i]

    #print sum(count_pos.values()), sum(count_neg.values())
    #print count_pos.most_common(10), count_neg.most_common(10)

    countTrain = list(count_posTrain) + list(count_negTrain)
    # change to dictionary as suggested by Ankit, original structure was having bad input format errors
    wordDict = {}
    for i in range(len(countTrain)):
        wordDict[countTrain[i]] = i

    #tempPos = count_posTest
    #tempNeg = count_negTest

    #for i in list(tempPos):

    #    if (tempPos[i] < 2*tempNeg[i] and tempNeg[i] < 2*tempPos[i]):
    #        del count_posTest[i]
    #        del count_negTest[i]

    #    if tempPos[i] < 2*tempNeg[i]:
    #        del count_posTest[i]

    #    if tempNeg[i] < 2*tempPos[i]:
    #        del count_negTest[i]


    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.

    train_pos_vec = []

    for sent in train_pos:
        temp = [0]*len(countTrain)
        for i in countTrain:
            if i in sent:
                temp[wordDict[i]] = 1
        train_pos_vec.append(list(temp))

    test_pos_vec = []

    for sent in test_pos:
        temp =[0]*len(countTrain)
        for i in countTrain:
            if i in sent:
                temp[wordDict[i]] = 1
        test_pos_vec.append(list(temp))

    train_neg_vec = []

    for sent in train_neg:
        temp =[0]*len(countTrain)
        for i in list(countTrain):
            if i in sent:
                temp[wordDict[i]] = 1
        train_neg_vec.append(list(temp))

    test_neg_vec = []

    for sent in test_neg:
        temp =[0]*len(countTrain)
        for i in list(countTrain):
            if i in sent:
                temp[wordDict[i]] = 1
        test_neg_vec.append(list(temp))


    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.

    labeled_train_pos =[]
    for i in range(len(train_pos)):
        string = "TRAIN_POS_" + str(i)
        sentence = LabeledSentence(train_pos[i],[string])
        labeled_train_pos.append(sentence)

    labeled_train_neg =[]
    for i in range(len(train_neg)):
        string = "TRAIN_NEG_" + str(i)
        sentence = LabeledSentence(train_neg[i],[string])
        labeled_train_neg.append(sentence)

    labeled_test_pos =[]
    for i in range(len(test_pos)):
        string = "TEST_POS_" + str(i)
        sentence = LabeledSentence(test_pos[i],[string])
        labeled_test_pos.append(sentence)

    labeled_test_neg =[]
    for i in range(len(test_neg)):
        string = "TEST_NEG_" + str(i)
        sentence = LabeledSentence(test_neg[i],[string])
        labeled_test_neg.append(sentence)

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run

    #most tests run on 1 instead of 5. Only final test run on 5.
    #hence final accuracies may have slight variations
    
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data

    train_pos_vec = []
    for i in range(len(train_pos)):
        string = "TRAIN_POS_" + str(i)
        train_pos_vec.append(model.docvecs[string])

    train_neg_vec = []
    for i in range(len(train_neg)):
        string = "TRAIN_NEG_" + str(i)
        train_neg_vec.append(model.docvecs[string])

    test_pos_vec = []
    for i in range(len(test_pos)):
        string = "TEST_POS_" + str(i)
        test_pos_vec.append(model.docvecs[string])

    test_neg_vec = []
    for i in range(len(test_neg)):
        string = "TEST_NEG_" + str(i)
        test_neg_vec.append(model.docvecs[string])

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    X = train_pos_vec + train_neg_vec
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    #X.reshape(len(train_pos_vec) + len(train_neg_vec), 1)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters

    nb_temp = sklearn.naive_bayes.BernoulliNB(alpha=1.0,binarize=None)
    nb_model = nb_temp.fit(X,Y)

    lr_temp = sklearn.linear_model.LogisticRegression()
    lr_model = lr_temp.fit(X,Y)

    return nb_model, lr_model


def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    X = train_pos_vec + train_neg_vec
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters

    nb_temp = sklearn.naive_bayes.GaussianNB()
    nb_model = nb_temp.fit(X,Y)

    lr_temp = sklearn.linear_model.LogisticRegression()
    lr_model = lr_temp.fit(X,Y)

    return nb_model, lr_model


def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.

    tp=tn=fn=fp=0
    X = test_pos_vec + test_neg_vec
    result = model.predict(X)

    for i in result[:len(test_pos_vec)]:
        if i == "pos":
            tp+=1
        else:
            fn+=1

    for i in result[len(test_pos_vec):]:
        if i == "pos":
            fp+=1
        else:
            tn+=1

    accuracy = float(tp + tn)/float(tp+tn+fp+fn)

    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
