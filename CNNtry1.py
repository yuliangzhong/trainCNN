# baseline
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, concatenate, BatchNormalization
from keras import Input
from keras.utils.vis_utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import argparse

def weighted_mse(y_true,y_pred):
    return K.mean(y_true/300*K.square(y_true - y_pred))

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=str, default='1', help='model name')
args = parser.parse_args()
modelName = args.n

data = np.load('sample_0211.npy')
y = data[:,0:2]
x = data[:,2:631]

xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15) 
xPostrain = xtrain[:,0:4]
xMaptrain = xtrain[:,4:629]
xPostest = xtest[:,0:4]
xMaptest = xtest[:,4:629]
xMaptrain = xMaptrain.reshape(xMaptrain.shape[0], 25, -1)

# Define different model structure here
Pos_input = Input(shape=(4),name='pose_input')
angLayer = Dense(256, activation='relu')(Pos_input)
angLayer = BatchNormalization()(angLayer)
out_ang = Dense(16, activation='relu')(angLayer)

Map_input = Input(shape=(25,25,1), name='map_input')
Layer = Conv2D(16, 5, activation="relu")(Map_input)
Layer = Conv2D(16, 5, activation="relu")(Layer)
Layer = MaxPooling2D((2,2))(Layer)
Layer = Conv2D(8, 5, activation="relu")(Layer)
Layer = Flatten()(Layer)
Layer = Dense(256, activation='relu')(Layer)
Layer = BatchNormalization()(Layer)
Layer = Dense(256, activation='relu')(Layer)
Layer = BatchNormalization()(Layer)
out_map = Dense(16, activation='relu')(Layer)

concatenated = concatenate([out_ang, out_map])
Layer2 = Dense(256, activation='relu')(concatenated)
Layer2 = BatchNormalization()(Layer2)
Layer2 = Dense(256, activation='relu')(Layer2)
Layer2 = BatchNormalization()(Layer2)
out = Dense(2, activation='relu')(Layer2)
model = Model([Pos_input, Map_input], out)
print(model.summary())
model.compile(loss='mse', optimizer="adam")
# plot_model(model, to_file="modelFig/model"+modelName+".png", show_shapes=False, show_dtype=False, show_layer_names=True, rankdir='TB', expand_nested=False)

# # fit and predict
model.fit([xPostrain, xMaptrain], ytrain, batch_size=16,epochs=300)
xMaptest = xMaptest.reshape(xMaptest.shape[0], 25, -1)
ypred = model.predict([xPostest,xMaptest]) # [v,g]

# save result
MSE1 = mean_squared_error(ytest[:,0], ypred[:,0])
MSE2 = mean_squared_error(ytest[:,1], ypred[:,1])

# define axis for subplots
plt.figure(figsize=(12,6))
axis1 = plt.subplot(1, 2, 1)
axis2 = plt.subplot(1, 2, 2)

rang = np.linspace(0,700,700)
axis1.scatter(ypred[:,0], ytest[:,0], s=5, color="blue")
axis1.scatter(rang,rang,s=3,color='red')
axis1.text(0,700,"Mean square error of new voxels: "+str(int(MSE1)))
axis1.set_xlabel("predicted new voxels")
axis1.set_ylabel("true new voxels")
axis2.scatter(ypred[:,1], ytest[:,1], s=5, color="blue")
axis2.scatter(rang,rang,s=3,color='red')
axis2.text(0,700,"Mean square error of gain: "+str(int(MSE2)))
axis2.set_xlabel("predicted gain")
axis2.set_ylabel("true gain")

plt.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig("result/result"+modelName+".png",dpi=200)
np.save("predData/ypred"+modelName+".npy",ypred)
np.save("testData/ytest"+modelName+".npy",ytest)

model.save("modelSave/model"+modelName)
