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
	output_train = argv[1]

	try:
	    opts, args = getopt.getopt(argv, 'i:o:', ['input=', 'output='])
	except getopt.GetoptError:
	    sys.exit(1)
	for opt, arg in opts:
	    if opt in ("-i", "--input"):
		input_dataset = arg
	    elif opt in ("-o", "--output"):
		output_train = arg
	
	print 'Input dataset: ' + input_dataset
	print 'Output train: ' + output_train


	data = pd.read_csv(
	    input_dataset,
	    parse_dates=["ts", "te"],  
	    index_col="ts",  
	    usecols=[  
		'ts', 'te', 'td', 'sa', 'da', 'sp', 'dp', 'pr', 'flg', 'ipkt', 'ibyt'],
	    engine = "c",  
	    dtype = {  
		'ts': 'str',
		'te': 'str',
		'td': 'float',
		'sa': 'str',
		'da': 'str',
		'sp': 'int',
		'dp': 'int',
		'pr': 'str',
		'flg': 'str',
		'ipkt': 'int',
		'ibyt': 'int',
	    },
	)



	print("Initial dataset size", data.shape)

	data.sort_index(kind="mergesort", inplace=True)
	data["te"] = data.index + pd.to_timedelta(data["td"], unit="s")

	#######

	pipeline = Pipeline([
	    ('Change_value_td', transformers.Change_value('td',0.0,0.0005)),
	    ('Insert_pps', transformers.Divider('ipkt','td','pps')),
	    ('Insert_bps', transformers.Divider('ibyt','td','bps')),
	    ('Insert_bpp', transformers.Divider('ipkt','ibyt','bpp')),
	    ('One_hot_encoder_pr', transformers.One_hot_enconder('pr')),
	    ('One_hot_enconder_flg', transformers.One_hot_enconder('flg'))
		             ])

	data2 = pipeline.transform(data)

	########

	data2 = data2.reset_index()[["ts", "td", "sa", "da", "sp", "dp" , "pr", "flg", "ipkt", "ibyt", "pps", "bps", "bpp"]]


	traf_sum= data2.groupby(by=["ts", "sa", "dp" , "pr", "flg"])["td","ipkt", "ibyt", "pps", "bps", "bpp"].sum() 
	traf_mean= data2.groupby(by=["ts", "sa", "dp" , "pr", "flg"])["td","ipkt", "ibyt", "pps", "bps", "bpp"].mean() 
	traf_median= data2.groupby(by=["ts", "sa", "dp" , "pr", "flg"])["td","ipkt", "ibyt", "pps", "bps", "bpp"].median() 
	traf_std= data2.groupby(by=["ts", "sa", "dp" , "pr", "flg"])["td","ipkt", "ibyt", "pps", "bps", "bpp"].std() 

	nconexiones= data2.groupby(by=["ts", "sa", "dp", "pr", "flg"]).size()
	traf_sum["nconexiones"] = nconexiones

	traffic = traf_sum.reset_index(("dp", "pr", "flg"))


	datanor = functions.normalize(traffic)
	datanor = datanor.fillna(value=0)
	print("Precessed dataset size",datanor.shape)
	#datanor.to_csv("dat_proc.csv")

	#datanor_train, datanor_test = funciones_ivan.descomposicion(datanor,datanor,2)
	#datanor = datanor.drop(['td','dp','pr','ipkt','ibyt','bpp'],axis=1)
	datanortrain = datanor.iloc[0:30000]

	data_test = datanor.values
	data_train = datanortrain.values

	alg = svm.OneClassSVM(nu=0.01)
	alg.fit(data_train)

	#alg = IsolationForest(n_estimators=100, contamination=0.01)
	#alg.fit(data_train)

	#alg = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
	#alg.fit(data_train)
	joblib.dump(alg, output_train)
	print("Training model exported correctly")






if __name__ == "__main__":
  main(sys.argv[1:])

