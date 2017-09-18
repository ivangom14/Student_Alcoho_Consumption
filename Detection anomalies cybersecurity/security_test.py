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
from sklearn import metrics
#from sklearn.neighbors import LocalOutlierFactor

def main (argv):
	input_dataset = argv[0]
	model_train = argv[1]
	ips = argv[2]
	dates = argv[3]

	try:
	    opts, args = getopt.getopt(argv, 'i:m:a:d:', ['input=', 'model=', 'address=', 'dates='])
	except getopt.GetoptError:
	    sys.exit(1)
	for opt, arg in opts:
	    if opt in ("-i", "--input"):
		input_dataset = arg
	    elif opt in ("-m", "--model"):
		model_train = arg
	    elif opt in ("-a", "--address"):
		ips = arg
	    elif opt in ("-d", "--dates"):
		dates = arg
	
	print 'Input dataset: ' + input_dataset
	print 'Output train: ' + model_train
	print 'IPs: ' + ips
	print 'Dates: ' + dates
	

	

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
	
	ips_txt = file(ips, "r")
	ips = ips_txt.read().splitlines()

	dates_txt = file(dates, "r")
	dates = dates_txt.read().splitlines()	

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

	#traf_sum= data2.groupby(by=["ts", "sa", "dp" , "pr", "flg"])["td","ipkt", "ibyt", "pps", "bps", "bpp"].sum
	traf_sum= data2.groupby(by=["ts", "sa", "dp" , "pr", "flg"])["td","ipkt", "ibyt", "pps", "bps", "bpp"].sum() 
	traf_mean= data2.groupby(by=["ts", "sa", "dp" , "pr", "flg"])["td","ipkt", "ibyt", "pps", "bps", "bpp"].mean() 
	traf_median= data2.groupby(by=["ts", "sa", "dp" , "pr", "flg"])["td","ipkt", "ibyt", "pps", "bps", "bpp"].median()  
	traf_std= data2.groupby(by=["ts", "sa", "dp" , "pr", "flg"])["td","ipkt", "ibyt", "pps", "bps", "bpp"].std() 


	nconexiones= data2.groupby(by=["ts", "sa", "dp", "pr", "flg"]).size()
	traf_sum["nconexiones"] = nconexiones

	traffic = traf_sum.reset_index(("ts", "sa", "dp", "pr", "flg"))
	

        #ip_source = ip_source.split(",")

	for i in ips:
	    traffic.loc[((traffic["sa"] == i)) & (traffic["ts"] > dates[0]) & (traffic["ts"] < dates[1]), 'anomaly'] = "yes"


	traffic = traffic.fillna(value="no")
	traffic = traffic.set_index(["ts", "sa", "anomaly"])
	traffic.to_csv("dat_processed.csv")
	datanor = functions.normalize(traffic)
	datanor = datanor.fillna(value=0)
	print("Precessed dataset size",datanor.shape)
	

	#datanor_train, datanor_test = functions.descomposicion(datanor,datanor,2)
	#datanor = datanor.drop(['td','dp','pr','ipkt','ibyt','bpp'],axis=1)
	

	data_test = datanor.values
	
	alg = joblib.load(model_train)
	
	prediction = alg.predict(data_test)
	#prediction = alg.fit_predict(data_test)

	hits = alg.decision_function(data_test)
	#hits = alg._decision_function(data_test)

	datanor["pred"] = prediction
	datanor["score"] = hits


	anomalies = datanor.pred.value_counts()
	#print(anomalies)

	res = pd.DataFrame()
	res['td_sum'] = traffic['td']
	res['dp'] = traffic['dp']
	res['pr'] = traffic['pr']
	res['ibyt_sum'] = traffic['ibyt']
	res['ipkt_sum'] = traffic['ipkt']
	res['bpp_sum'] = traffic['bpp']
	res['nconexiones'] = traffic['nconexiones']
	res['score'] = datanor['score']
	res['pred'] = datanor['pred']

	res2 = res[res['pred']==-1].sort_values(by='score', ascending=True)
	res3 = res2.drop(['pred'], axis=1)
	res3.to_csv("results.csv")
	print("Results exported correctly")
	#print(res3)

	anom = res3.reset_index(("anomaly"))
	TP = anom[(anom['anomaly'] == "yes")]
	TP = TP['anomaly'].count()
	FP = anom[(anom['anomaly'] == "no")]
	FP = FP['anomaly'].count()

	res4 = res[res['pred']==1].sort_values(by='score', ascending=True)
	noanom = res4.reset_index(("anomaly"))
	TN = noanom[(noanom['anomaly'] == "no")]
	TN = TN['anomaly'].count()
	FN = noanom[(noanom['anomaly'] == "yes")]
	FN = FN['anomaly'].count()

	PPV = float(TP)/(TP+FP)
	print("PPV (Precision):", PPV)
	TPR = float(TP)/(TP+FN)
	print("TPR (Recall):", TPR)
	F1 = 2*((PPV*TPR)/(PPV+TPR))
	print("F1 (Score):", F1)
	ACC = (float(TP)+TN)/(TP+TN+FP+FN)
	print("ACC (Accuracy):", ACC)
	TNR = float(TN)/(TN+FP)
	print("TNR (Specificity):", TNR)
	FPR = 1-TNR
	print("FPR (Fall out):", PPV)
	AUC = (TPR+TNR)/2
	print("AUC :", AUC)



if __name__ == "__main__":
  main(sys.argv[1:])

