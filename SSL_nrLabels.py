import numpy as np
import pandas as pd
import random
import pdb
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from scipy import sparse
from plotFunctions import surface
from helper import createConfigName, generateVocabulary
from itertools import product
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

def SSL_nrLabels():
	# PARAMETERS
	RBF = 1
	gammaArray = [0.25, 0.5, 1, 5, 10]
	gammaArray = [15,20,50]
	gammaArray = [1,10]
	gamma = 5
	conversion = 'tf_tfidf'
	nrLabeledData = [1,2,5,10,15,50,100,250]
	nrLabeledData = [1,5,10]
	featureConv = None
	cosSim = 1
	
	# Load Data
	name = 'NG_rec'
	path = '/vol/tensusers/nwidmann/'
	filename = path + 'processedDocuments/' + name + '.pkl'
	resultFilename = path + 'nrLabels/' + name + '_test_' + str(conversion) + '_cos'
	results = dict()
	

	data = pd.read_pickle(filename)
	vocabulary = generateVocabulary(data.sentences.tolist())
	X_org = np.load(path + 'matrices/' + name + '.npy')
	X_org = X_org[:-2,:-2]
	nrDocs = len(data)

	#contextSim_org = np.load(path + 'matrices/' + name + '_contextSim.npy')
	#contextSim_org = contextSim_org[:-2,:-2]
	
	FF = X_org[nrDocs:, nrDocs:]
        ngram = np.dot(FF, FF)
	contextSim_org = ngram

	# remove DD and FD
	#X = X[:,nrDocs:]
	# remove $Start$ and $End$

	for run in range(2):
		print 'RUN ' + str(run)

		X = np.copy(X_org)
		contextSim = np.copy(contextSim_org)

		if featureConv == 'ngram':
			FF = X[nrDocs:, nrDocs:]
			ngram = np.dot(FF, FF)
        	        X[nrDocs:, nrDocs:] = (ngram+FF)/2

		if featureConv == 'contextSim':
			X[nrDocs:, nrDocs:] = contextSim
			

		if featureConv == 'avgFF':
			FF = np.add(X[nrDocs:, nrDocs:], contextSim)/2
			X[nrDocs:, nrDocs:] = FF 

		# Renormalize
		#if renormalize:
		#	DF = X[nrDocs:, :nrDocs]
		#	rowsums = DF.sum(axis=1) 
		#	for i in range(len(rowsums)):
		#		DF[i] = DF[i]/rowsums[i]
		#	X[nrDocs:,:nrDocs] = DF
		#	X[:nrDocs,nrDocs:] = np.transpose(DF)
		#FF = X[nrDocs:,:]
		#X[nrDocs:,:] = np.transpose(FF)
		
		# Remove posts with no features
		#DF = X[:nrDocs,:]
		#indZeroFeatures = np.where(DF.sum(axis=1)==0)[0]
		#for ind in indZeroFeatures:
		#	X = np.delete(X,ind,0)
		#data.drop(data.index[indZeroFeatures], inplace=True)
		#data.index = range(len(data)) 
		#nrDocs = len(data)
		
		# Normalize
		#DF = X[:nrDocs,:] 
		#FF = X[nrDocs:,:]
		#rowsum = DF.sum(axis=1)
		#X[nrDocs:, nrDocs:] = np.transpose(X[nrDocs:, nrDocs:])
		#X[-1,-1] = 1
		#FF = X[nrDocs:, nrDocs:]
		#FF_rowsum = FF.sum(axis=1)

		if conversion=='tfidf':
			DF = np.array(data.tfIdf.tolist())
			X[:nrDocs, nrDocs:] = DF
			X[nrDocs:, :nrDocs] = np.transpose(DF)

		if conversion=='tf_tfidf':
        		DF = np.array(data.tfIdf.tolist())
        		X[:nrDocs, nrDocs:] = DF

		if conversion=='raw_tfidf':
			DF = np.array(data.tfIdf.tolist())
			X = DF

		if conversion=='MM':
			DF = X[:nrDocs, nrDocs:]
			FF = X[nrDocs:, nrDocs:]
			X = np.dot(DF,FF)

		if cosSim:
			X = cosine_similarity(X,X)


		curr_results = pd.DataFrame(data = {'nrLabels': nrLabeledData})
		conditions = ['LP_prec', 'LP_rec', 'LP_fscore', 'LS_prec', 'LS_rec', 'LS_fscore', 'tfIdf_LP_prec', 'tfIdf_LP_rec', 'tfIdf_LP_fscore', 'tfIdf_LS_prec', 'tfIdf_LS_rec', 'tfIdf_LS_fscore']
		for col in product(conditions, gammaArray):
			colName = col[0]+'_'+str(col[1])
			curr_results[colName] = 0 
		baseline =  ['SVC_prec', 'SVC_rec', 'SVC_fscore']
		for col in baseline:
			curr_results[col] = 0
		
		
        	for indNr,nrLabels in enumerate(nrLabeledData):
			print 'Number Labels: ' + str(nrLabels)

			docLabels = np.ones(nrDocs)*-1
			for cat in data.category.unique():
				indices = data[data.category==cat].index
				rndSample = random.sample(indices, nrLabels)
				docLabels[rndSample] = cat
			unlabeledDocs = np.where(docLabels==-1)[0]
			testLabels = data.loc[unlabeledDocs, 'category'].tolist()
			
			nrFeatures = X.shape[0] - nrDocs
			featureLabels = np.ones(nrFeatures)*-1
			labels = np.hstack((docLabels, featureLabels))

			#X[nrDocs:, nrDocs:] = rbf_kernel(X[nrDocs:, nrDocs:], X[nrDocs:, nrDocs:], gamma=0.5)
			
			if RBF:
				labelPropagation = LabelPropagation('rbf', gamma=gamma, alpha=1, useInputMatrix=0, max_iter=20)
				labelPropagation.fit(X, labels)
				predictLabels = labelPropagation.transduction_
				evalMeasures = precision_recall_fscore_support(testLabels, predictLabels[unlabeledDocs], average='macro')
				name = [elem+'_'+str(gamma) for elem in ['LP_prec', 'LP_rec', 'LP_fscore']]
				curr_results.loc[indNr, name] = evalMeasures[0:3]
				# Label Spread
				labelSpread = LabelSpreading('rbf', gamma=gamma, alpha=1, max_iter=20)
				labelSpread.fit(X,labels)
				predictLabels = labelSpread.transduction_
				evalMeasures = precision_recall_fscore_support(testLabels, predictLabels[unlabeledDocs], average='macro')	
				name = [elem+'_'+ str(gamma) for elem in ['LS_rec', 'LS_rec', 'LS_fscore']]
				curr_results.loc[indNr, name] = evalMeasures[0:3] 
				
			else:
				labelPropagation = LabelPropagation(alpha=1, useInputMatrix=1, max_iter=200)
				print labelPropagation
				labelPropagation.fit(X, labels)
				predictLabels = labelPropagation.transduction_
				print 'Test Accuracy: %f' % accuracy_score(data.category.tolist()[nrLabeledData+1:], predictLabels.tolist()[nrLabeledData+1:len(data)])


				
				DF = np.array(data.tfIdf.tolist())

				# tfIdf SSL baseline
				labelPropagation = LabelPropagation('rbf', gamma=gamma, alpha=1, useInputMatrix=0, max_iter=100)
				labelPropagation.fit(DF, docLabels)
				predictLabels = labelPropagation.transduction_
				curr_acc = accuracy_score(testLabels, predictLabels[unlabeledDocs])
				evalMeasures = precision_recall_fscore_support(testLabels, predictLabels[unlabeledDocs], average='macro')
				name = [elem+'_'+str(gamma) for elem in ['tfIdf_LP_prec', 'tfIdf_LP_rec', 'tfIdf_LP_fscore']]
				curr_results.loc[indNr, name] = evalMeasures[0:3]

				labelSpread = LabelSpreading('rbf', gamma=gamma, alpha=1)
				labelSpread.fit(DF,docLabels)
				predictLabels = labelSpread.transduction_
				
				evalMeasures = precision_recall_fscore_support(testLabels, predictLabels[unlabeledDocs], average='macro')
				name = [elem+'_'+str(gamma) for elem in ['tfIdf_LS_prec', 'tfIdf_LS_rec', 'tfIdf_LS_fscore']]
                                curr_results.loc[indNr, name] = evalMeasures[0:3]

				# TF-IDF supervised Baseline
				clf = MultinomialNB()
				clf = SVC()
				indLabeled = np.where(docLabels!=-1)[0].tolist()
				features = data.loc[indLabeled, 'tfIdf'].tolist()
        	                labels = data.loc[indLabeled, 'category'].tolist()

				clf.fit(features, labels)
				pred = clf.predict(data.loc[unlabeledDocs, 'tfIdf'].tolist())
				
				evalMeasures = precision_recall_fscore_support(testLabels, predictLabels[unlabeledDocs], average='macro')
                                curr_results.loc[indNr, baseline] = evalMeasures[0:3]

		results['Run'+str(run)] = curr_results
		
		
	resultsPanel = pd.Panel.from_dict(results, orient='minor')
	pdb.set_trace()
	means = resultsPanel.mean(axis=2)
	std = resultsPanel.std(axis=2)
	means.to_csv(resultFilename + '_mean.csv', sep='\t', encoding='utf-8')
	std.to_csv(resultFilename + '_std.csv', sep='\t', encoding='utf-8')


if __name__ =='__main__':
	SSL_nrLabels()
