from __future__ import division
import sys, os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
from sklearn.utils import shuffle
import re
import string
import math
from ROOT import TFile, TTree
from ROOT import *
import ROOT
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Dropout, add
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from array import array
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from keras.utils.np_utils import to_categorical
from keras import optimizers

timer = ROOT.TStopwatch()
timer.Start()

trainInput = "/home/juhee5819/T2+/ttbb_pt20.h5"

data = pd.read_hdf(trainInput)

#make the number of events uniformly
cat_num_list = [(len(data.loc[data['category'] == i])) for i in range(7)]
smallest = min(cat_num_list)
print cat_num_list, smallest
#print cat_num_list

pd_droped = pd.DataFrame()
np.random.seed(10)
for i in range(7):
	pd_cat = data.loc[data['category'] == i]
	remove_cat = len(data.loc[data['category'] == i]) - smallest
	cat_drop_indices = np.random.choice(pd_cat.index, remove_cat, replace=False)
	pd_cat_droped = pd_cat.drop(cat_drop_indices)
	pd_droped = pd_droped.append(pd_cat_droped)
	print len(pd_cat_droped)

# pickup only interesting variables
variables = ["jet1_pt", "jet1_eta", "jet1_phi", "jet1_e", "jet1_btag", "jet1_CvsB", "jet1_CvsL", "jet2_pt", "jet2_eta", "jet2_phi", "jet2_e", "jet2_btag", "jet2_CvsB", "jet2_CvsL", "jet3_pt", "jet3_eta", "jet3_phi", "jet3_e", "jet3_btag", "jet3_CvsB", "jet3_CvsL", "jet4_pt", "jet4_eta", "jet4_phi", "jet4_e", "jet4_btag", "jet4_CvsB", "jet4_CvsL", "dR12", "dR13", "dR14", "dR23", "dR24", "dR34", "dRlep1", "dRlep2", "dRlep3", "dRlep4", "dEta12", "dEta13", "dEta14", "dEta23", "dEta24", "dEta34", "dPhi12", "dPhi13", "dPhi14", "dPhi23", "dPhi24", "dPhi34", "invm12", "invm13", "invm14", "invm23", "invm24", "invm34", "invmlep1", "invmlep2", "invmlep3", "invmlep4"]

pd_train_out = pd_droped.filter(items = ['category'])
pd_train_data = pd_droped.filter(items = variables)

#covert from pandas to array
train_out = np.array( pd_train_out )
train_data = np.array( pd_train_data )

numbertr=len(train_out)

trainnb=0.7 # Fraction used for training

#Shuffling
order=shuffle(range(numbertr),random_state=100)
train_out=train_out[order]
train_data=train_data[order,0::]

#Splitting between training set and cross-validation set
pd_valid = data[int(trainnb*numbertr):numbertr]
pd_train = data[0:int(trainnb*numbertr)]

#Splitting between training set and cross-validation set
valid_data=train_data[int(trainnb*numbertr):numbertr,0::]
valid_data_out=train_out[int(trainnb*numbertr):numbertr]
valid_data_out = to_categorical(valid_data_out)

train_data_out=train_out[0:int(trainnb*numbertr)]
train_data=train_data[0:int(trainnb*numbertr),0::]
train_data_out = to_categorical(train_data_out)

import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(20, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(20, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(20, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(50, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(100, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(300, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(300, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(300, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(100, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(100, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(7, activation=tf.nn.softmax)
])

#modelshape = "10L_300N"
batch_size = 256
epochs = 100
model_output_name = 'model_ttbar_%dE' %(epochs)

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy', 'categorical_accuracy'])
hist = model.fit(train_data, train_data_out, batch_size=batch_size, epochs=epochs, validation_data=(valid_data,valid_data_out))

    #using only fraction of data
    #evaluate = model.predict( valid_data ) 

model.summary()

pred = model.predict(valid_data)
pred = np.argmax(pred, axis=1)
comp = np.argmax(valid_data_out, axis=1)

print len(valid_data)

#result = pd.DataFrame({"real":comp, "pred":pred, "btagcondi":btagcondi})
result = pd.DataFrame({"real":comp, "pred":pred})
#result.drop(result[result(['btagcondi'] == 0)].index, inplace = True) 

result_array = []
for i in range(7):
    result_real = result.loc[result['real']==i]
    temp = [(len(result_real.loc[result_real["pred"]==j])) for j in range(7)]
    result_array.append(temp)
    print temp, len(result_real)

result_array_prob = []
for i in range(7):
    result_real = result.loc[result['real']==i]
    temp = [(len(result_real.loc[result_real["pred"]==j])) / len(result_real) for j in range(7)]
    result_array_prob.append(temp)
    print temp, len(result_real)
print result_array_prob
#np.savetxt("result.csv", result_array + result_array_prob )

#Heatmap
plt.rcParams['figure.figsize'] = [7.5, 6]
cfmt = lambda x,pos: '{:.0%}'.format(x)
heatmap = sns.heatmap(result_array_prob, annot=True, cmap='YlGnBu', fmt='.1%', annot_kws={"size":12}, vmax=1, cbar_kws={'format': FuncFormatter(cfmt)} )
#plt.title('Heatmap', fontsize=15)
plt.xlabel('pred.', fontsize=12)
plt.ylabel('real', fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
plt.show()

print("Plotting scores")
plt.plot(hist.history['categorical_accuracy'])
plt.plot(hist.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Test'], loc='lower right')
plt.savefig(os.path.join('fig_score_acc.pdf'))
plt.gcf().clear()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc='upper right')
#plt.savefig(os.path.join('fig_score_loss.pdf'))
plt.savefig(os.path.join('fig_score_loss_ctag.pdf'))
plt.gcf().clear()

timer.Stop()
rtime = timer.RealTime(); # Real time (or "wall time")
ctime = timer.CpuTime(); # CPU time
print("RealTime={0:6.2f} seconds, CpuTime={1:6.2f} seconds").format(rtime,ctime)
