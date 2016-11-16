import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics, preprocessing
import numpy as np
import pandas as pd

df = pd.read_csv("data/CHD.csv", header=0)
print df.describe()

def my_model(X, y):
    return skflow.models.logistic_regression(X, y)

a = preprocessing.StandardScaler()

X =a.fit_transform(df['age'].astype(float))

print a.get_params()
classifier = skflow.TensorFlowEstimator(model_fn=my_model, n_classes=1)
classifier.fit(X, df['chd'].astype(float), logdir='/tmp/logistic')
print(classifier.get_tensor_value('logistic_regression/bias:0'))
print(classifier.get_tensor_value('logistic_regression/weight:0'))
score = metrics.accuracy_score(df['chd'].astype(float), classifier.predict(X))
print("Accuracy: %f" % score)

