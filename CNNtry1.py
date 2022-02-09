# baseline
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, concatenate, BatchNormalization
from keras import Input
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=str, default='1', help='model number')
args = parser.parse_args()
modelNo = args.n

data = np.load('sample_0207.npy')
y = data[:,0]
x = data[:,1:628]
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15) 
xAngtrain = xtrain[:,0:2]
xMaptrain = xtrain[:,2:628]
xAngtest = xtest[:,0:2]
xMaptest = xtest[:,2:628]
xMaptrain = xMaptrain.reshape(xMaptrain.shape[0], 25, -1)

# Define different model structure here
Ang_input = Input(shape=(2),name='angle_input')
angLayer = Dense(128, activation='relu')(Ang_input)
angLayer = Dense(64, activation='relu')(angLayer)
out_ang = Dense(16, activation='relu')(angLayer)

Map_input = Input(shape=(25,25,1), name='map_input')
Layer = Conv2D(16, 5, activation="relu")(Map_input)
Layer = Conv2D(16, 5, activation="relu")(Layer)
Layer = MaxPooling2D((2,2))(Layer)
Layer = Conv2D(8, 5, activation="relu")(Layer)
Layer = Flatten()(Layer)
Layer = Dense(128, activation='relu')(Layer)
Layer = Dense(64, activation='relu')(Layer)
out_map = Dense(16, activation='relu')(Layer)

concatenated = concatenate([out_ang, out_map])
Layer2 = Dense(128, activation='relu')(concatenated)
Layer2 = Dense(64, activation='relu')(Layer2)
out = Dense(1, activation='relu')(Layer2)
model = Model([Ang_input, Map_input], out)
print(model.summary())
model.compile(loss='mse', optimizer="adam")
# plot_model(model, to_file="modelFig/model"+modelNo+".png", show_shapes=False, show_dtype=False, show_layer_names=True, rankdir='TB', expand_nested=False)

# fit and predict
model.fit([xAngtrain, xMaptrain], ytrain, batch_size=32,epochs=200)
xMaptest = xMaptest.reshape(xMaptest.shape[0], 25, -1)
ypred = model.predict([xAngtest,xMaptest])

# save result
MSE = mean_squared_error(ytest, ypred)

rang = np.linspace(0,700,700)
plt.scatter(ypred, ytest, s=5, color="blue")
plt.scatter(rang,rang,s=5,color='red')
plt.text(0,700,MSE)
plt.savefig("result/result"+modelNo+".png")
np.save("predData/ypred"+modelNo+".npy",ypred)
np.save("testData/ytest"+modelNo+".npy",ytest)

model.save("modelSave/model"+modelNo)
