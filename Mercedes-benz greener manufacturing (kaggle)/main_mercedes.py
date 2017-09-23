import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn import decomposition
import matplotlib.pyplot as plt
import transformers
from sklearn.pipeline import Pipeline

datatest = pd.read_csv(
    "data/test.csv",
    usecols=[  
        'ID','X0', 'X1', 'X2', 'X3', 'X4',	'X5', 'X6', 'X8', 'X10','X314',"X47","X118","X315","X29","X127","X236","X115","X383","X152","X342","X151","X351","X327","X77","X104","X267","X95","X142","X58","X116","X27","X273","X339","X204","X261","X177","X70","X46","X240","X50","X133","X316","X313","X64","X68","X71","X350","X225","X292","X96","X345","X164",
    ],
    engine = "c",  
)

datatrain = pd.read_csv(
    "data/train.csv",
    index_col=["y"],  
    usecols=[  
         'y', 'ID', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8', 'X10','X314',"X47","X118","X315","X29","X127","X236","X115","X383","X152","X342","X151","X351","X327","X77","X104","X267","X95","X142","X58","X116","X27","X273","X339","X204","X261","X177","X70","X46","X240","X50","X133","X316","X313","X64","X68","X71","X350","X225","X292","X96","X345","X164",
    ],
    engine = "c",  

)


pipeline = Pipeline([
    ('Drop_ID', transformers.Columns_dropper('ID')),
    ('One_hot_encoder_X0', transformers.One_hot_enconder('X0')),
    ('One_hot_encoder_X1', transformers.One_hot_enconder('X1')),
    ('One_hot_encoder_X2', transformers.One_hot_enconder('X2')),
    ('One_hot_encoder_X3', transformers.One_hot_enconder('X3')),
    ('One_hot_encoder_X4', transformers.One_hot_enconder('X4')),
    ('One_hot_encoder_X5', transformers.One_hot_enconder('X5')),
    ('One_hot_encoder_X6', transformers.One_hot_enconder('X6')),
    ('One_hot_enconder_X8', transformers.One_hot_enconder('X8')),
    ('Normalize', transformers.Normalizer()),
    ('Add_decomposition', transformers.Decomposition()),
])

datatest2 = pipeline.transform(datatest)
datatrain2 = pipeline.transform(datatrain)


datatest3 = datatest2.values
indextest3 = datatest2.index
datatrain3 = datatrain2.values
indextrain = datatrain2.index


alg = xgb.sklearn.XGBRegressor(max_depth=4,learning_rate=0.005, subsample=0.921, objective='reg:linear',n_estimators=1300)
alg2 = alg.fit(datatrain3, indextrain)
indexpred = alg2.predict(datatest3)

#fig, ax = plt.subplots(1,1, figsize=(8,64))
#xgb.plot_importance(alg2, height=0.5, ax=ax)
#plt.show()

sub = pd.DataFrame()
sub['ID'] = datatest['ID']
sub['y'] = indexpred
print(sub)
sub.to_csv("prediction.csv", index= False, float_format= '%.6f', decimal= '.')
