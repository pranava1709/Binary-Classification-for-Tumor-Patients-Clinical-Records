import pandas as pd 
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
data = pd.read_csv("C:/OHSL/UCSF-PDGM-metadata.csv")
nan_values = data.isna()
data = data.dropna()
id_in = data['ID']
sez_in = data.iloc[:,1].index
gen_id = data.iloc[:,4:6].index
gen1_id = data.iloc[:,7:9].index
gen2_id = data.iloc[:,11:].index
data = data.drop('ID',axis = 1)
data = data.drop('Sex',axis = 1)
data = data.drop('Final pathologic diagnosis (WHO 2021)',axis = 1)
data = data.drop('IDH',axis = 1)
data = data.drop('EOR',axis = 1)
data = data.drop('Biopsy prior to imaging',axis = 1)
data = data.drop('MGMT status',axis = 1)
data = data.drop('1p/19q',axis = 1)

new = ['Age at MRI','WHO CNS Grade','MGMT index','OS','1-dead 0-alive']

data = data.reindex(columns= new)
correlation  = data.corr()
covariance = data.cov()
print(correlation)
print(covariance)
sns.heatmap(correlation)

plt.show()
#sns.heatmap(covariance)
#plt.show()

#print(data)
#data.to_csv("fin.csv")

train,test = train_test_split(data,test_size = 0.20,random_state = 0)
X_train  = train.iloc[:,0:4]
Y_train = train.iloc[:,4:5]
X_test = test.iloc[:,0:4]
Y_test = test.iloc[:,4:5]
Y_test1 = test.iloc[:,4:5]
#Y_test1 = Y_test1.astype("int")
X_train = np.array(X_train)
Y_train= np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(X_test)
np.savetxt("inf1.txt",Y_test1)
X_train = X_train.T
Y_train = Y_train.T
X_test =X_test.T
Y_test=Y_test.T
print(Y_test)


def weights(dim):
	w = np.zeros((dim, 1))
	b = 0.0
	return w,b
	w = np.squeeze(w,axis = 0)
	print(w.shape)

def pred(z):
	fin = 1/(1+np.exp(-z))
	return fin
def feed_forward(X_train,Y_train,w,b):
	z = np.dot(w.T,X_train)+b

	fin = pred(z)

	loss = -Y_train * np.log(fin)+ (1-Y_train)*( np.log(1-fin))
	cost = np.sum(loss)/X_train.shape[1] 
	
	return cost


def back_prop(X_train,Y_train,w,b):
	z = np.dot(w.T,X_train)+b
	fin  = pred(z)
	weight = np.dot(X_train,(fin-Y_train).T)/X_train.shape[1]
	bias = np.sum(fin-Y_train)/X_train.shape[1]
	gradients = {"weight":weight,"bias":bias}
	return gradients




def upd(X_train,Y_train,w,b,noi,lr):

	for i in range(noi):
		cost  = feed_forward(X_train,Y_train,w,b)
		gradients  = back_prop(X_train,Y_train,w,b)
		dw = gradients["weight"]
		db = gradients["bias"]
		w = w - lr* dw
		b = b - lr * db
		parameters= {"weights":w,"bias":b}
		if i % 5 == 0:
			print(cost)
	return gradients,parameters

def inf(w,b,X_test):
	z= pred(np.dot(w.T,X_test)+b)
	predictions = np.zeros((1,X_test.shape[1]))
	for j in range(X_test.shape[1]):
		if z[0,j]>=0.5:
			predictions[0,j] = 1
			#print(predictions )
		else:
			predictions[0,j] = 0
			#print(predictions)
	return predictions
	#print(predicions)
	

def lrr(X_train, Y_train, X_test, Y_test,noi,lr):
	dim = X_train.shape[0]

	w,b = weights(dim)
	#w = np.squeeze(w,axis = 1)
	#print(w.shape)

	gradient,parameters = upd(X_train,Y_train,w,b,noi,lr)
	predictions_test = inf(parameters["weights"],parameters["bias"],X_test)
	predictions_train = inf(parameters["weights"],parameters["bias"],X_train)
	
	pt = np.array(predictions_test)
	pt_df = pd.DataFrame(pt)
	pt1 = pt_df.iloc[0:1,:].T

	#np.savetxt("pt.txt",predictions_test)
	print(pt1)
	metric = f1_score(pt1,Y_test1,average = None)
	print(metric)
lrr(X_train, Y_train, X_test, Y_test, 100,0.1)

