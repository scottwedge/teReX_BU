import numpy as np
import pandas as pd
import random
import pdb
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from scipy import sparse
from plotFunctions import surface
from helper import createConfigName, generateVocabulary
from itertools import product
from wordcloud import WordCloud

def SSL():
	# PARAMETERS
	RBF = 1
	gammaArray = [0.01, 0.15, 0.25, 0.5, 0.75,1,2,5,10]
	gammaArray = [1,2,5]
	gammaArray = [5]
	convs = [None, 'tfidf', 'MM', 'raw_tfidf', 'tf_tfidf']
	conversion = convs[4] 
	nrLabeledData = 100 
	# Load Data
	name = 'NG_rec'
	path = '/vol/tensusers/nwidmann/'
	filename = path + 'processedDocuments/' + name + '.pkl'

	X_org = np.load(path+'matrices/' + name + '.npy')
	X_org = X_org[:-2,:-2]
        #contextSim_org = np.load(path+'matrices/' + name + '_contextSim.npy')
        
	data = pd.read_pickle(filename)
	vocabulary = generateVocabulary(data.sentences.tolist())
	invVocab = dict(zip(vocabulary.values(), vocabulary.keys()))
	nrDocs = len(data)

	resultFilename = path + 'results/' + name + '_fscore_final_' + str(conversion) + '_'+ str(nrLabeledData) 
	results = dict()

	pdb.set_trace()
	

	for loop in range(3):
		print loop
		
		curr_result = pd.DataFrame(data = {'gamma': gammaArray})
		docLabels = np.ones(nrDocs)*-1
		for cat in data.category.unique():
			indices = data[data.category==cat].index
			rndSample = random.sample(indices, nrLabeledData)
			docLabels[rndSample] = cat
        	unlabeledDocs = np.where(docLabels==-1)[0]
		testLabels = data.loc[unlabeledDocs, 'category'].tolist()

		#for params in product((0,1),('_', 'avgFF', 'contextSim')):
		#for params in product((0,1),('_')):
		for params in product((0,1),('_', 'avgFF', 'ngram')):
			params = (1,'_')
		#for dummyInd in range(1):
			cosSim = params[0]
			featureConv = params[1] 

			configName = createConfigName(cosSim, featureConv)
			print configName
		
			X = np.copy(X_org)
			#contextSim = np.copy(contextSim_org)
			#contextSim = contextSim[:-2,:-2]
			
			FF = X[nrDocs:, nrDocs:]
			ngram = np.dot(FF, FF)
			contextSim = ngram

			# remove DD and FD
			#X = X[:,nrDocs:]
			# remove $Start$ and $End$

			if featureConv == 'ngram':
				FF = X[nrDocs:, nrDocs:]
				ngram = np.dot(FF, FF)
        	                #X[nrDocs:, nrDocs:] = (ngram+FF)/2
				X[nrDocs:, nrDocs:] = ngram

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

			nrFeatures = X.shape[0] - nrDocs
			featureLabels = np.ones(nrFeatures)*-1
			labels = np.hstack((docLabels, featureLabels))

			#X[nrDocs:, nrDocs:] = rbf_kernel(X[nrDocs:, nrDocs:], X[nrDocs:, nrDocs:], gamma=0.5)
			
			if RBF:
				labelProp_accuracy = []
				labelSpread_accuracy = []

				for gamma in gammaArray:
					print gamma
					#m = rbf_kernel(X,X, gamma=gamma)
					labelPropagation = LabelPropagation('rbf', gamma=gamma, alpha=1, useInputMatrix=0, max_iter=20)
					labelPropagation.fit(X, labels)
					predictLabels = labelPropagation.transduction_
					prec_rec_fscore = precision_recall_fscore_support(testLabels, predictLabels[unlabeledDocs], average='macro')
					labelProbabilities = np.max(labelPropagation.label_distributions_, axis=1)

					ind = np.where(predictLabels==0)[0]
					wc = [(invVocab[ind], labelProbabilities[ind] for ind in indices]
                                        wc_c1 = 1 


					pdb.set_trace()
					#print confusion_matrix(testLabels, predictLabels[unlabeledDocs], labels=data.category.unique())
					labelProp_accuracy.append(prec_rec_fscore[0:3])
					
					
						
					labelSpread = LabelSpreading('rbf', gamma=gamma, alpha=1, max_iter=20)
					labelSpread.fit(X,labels)
					predictLabels = labelSpread.transduction_
					prec_rec_fscore = precision_recall_fscore_support(testLabels, predictLabels[unlabeledDocs], average='macro')
					#print confusion_matrix(testLabels, predictLabels[unlabeledDocs], labels=data.category.unique())
					labelSpread_accuracy.append(prec_rec_fscore[0:3])

				results_LP = pd.DataFrame(labelProp_accuracy, columns=['LP_prec'+configName,'LP_rec'+configName, 'LP_fscore'+configName])
				results_LS = pd.DataFrame(labelSpread_accuracy, columns=['LS_prec'+configName,'LS_rec'+configName , 'LS_fscore'+configName])
				curr_result = pd.concat([curr_result, results_LP, results_LS], axis=1)
				
				
			else:
				labelPropagation = LabelPropagation(alpha=1, useInputMatrix=1, max_iter=20)
				print labelPropagation
				labelPropagation.fit(X, labels)
				predictLabels = labelPropagation.transduction_
                                prec_rec_fscore = precision_recall_fscore_support(testLabels, predictLabels[unlabeledDocs], average='macro')
				res = pd.DataFrame([prec_rec_fscore[0:3]], columns=['LP_prec'+configName,'LP_rec'+configName, 'LP_fscore'+configName])
				curr_result = pd.concat([curr_result, res], axis=1)
		
		#results = pd.concat((results, curr_result), axis=1)
		#results = results.groupby(results.columns).mean()
		results['Run'+str(loop)] = curr_result
		print results

	results = pd.Panel.from_dict(results, orient='minor')
	meansEval = results.mean(axis=2)
	stdEval = results.std(axis=2)	
	#pdb.set_trace()
	meansEval.to_csv(resultFilename + '_mean.csv', sep='\t', encoding='utf-8')
	stdEval.to_csv(resultFilename + '_std.csv', sep='\t', encoding='utf-8')

	#pdb.set_trace()
	print 'End of script'


if __name__ =='__main__':
	SSL()
