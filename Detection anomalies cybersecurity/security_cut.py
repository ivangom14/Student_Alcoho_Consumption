import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.pipeline import Pipeline
import transformers
from sklearn import svm
import sys
from sklearn.externals import joblib
import getopt


def main (argv):
	input_dataset = argv[0]
	output_dataset = argv[1]

	try:
	    opts, args = getopt.getopt(argv, 'i:o:', ['input=', 'output='])
	except getopt.GetoptError:
	    sys.exit(1)
	for opt, arg in opts:
	    if opt in ("-i", "--input"):
		input_dataset = arg
	    elif opt in ("-o", "--output"):
		output_dataset = arg
	
	print 'Input dataset: ' + input_dataset
	print 'Output dataset: ' + output_dataset


	data = pd.read_csv(input_dataset)
	
	print("Dataset size",data.shape)
	#data.sort_index(kind="mergesort", inplace=True)
	#data2 = data[data['sa'] != '192.168.1.43']
	#data3 = data2[data['sa'] != '192.168.1.50']
	data3 = data.iloc[5000:10000]
	
	print("Cut dataset size ",data3.shape)
	data3.to_csv(output_dataset,index= False)

	print("Dataset cut exported correctly")

if __name__ == "__main__":
  main(sys.argv[1:])

