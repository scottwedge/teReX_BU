from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd 
import numpy as np
import pdb


def supervisedBaseline():

	nrLabels = 1 
	path = '/vol/tensusers/nwidmann/'
	filename= path + 'processedDocuments/NG_rec.pkl'
	data = pd.read_pickle(filename) 
	n = len(data)
        
	#features = data.CountVectors.tolist()
	features = data.tfIdf.tolist()
	labels = data.category.tolist()

	pdb.set_trace()

	clf = SVC()
	clf.fit(features[6:10], labels[6:10])
	pred = clf.predict(features[11:n])

	accuracy = accuracy_score(labels[11:n], pred)
	prec, recall, fscore, beta = precision_recall_fscore_support(labels[11:n], pred, average='macro')
	print 'Test Accuracy: %f' % accuracy 
	print 'Fscore: %f' % fscore
	


if __name__ == '__main__':
    supervisedBaseline()
