
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import keras

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense


cos = np.cos
pi = np.pi
sin = np.sin

# données du problème
#borne de l'intervalle
a = 0
b = 1

#condition initiale
V_0x = 1
V_0y = 0
V_0 = np.array([V_0x, V_0y])

#pulsation
W = 2*pi

#fonctions définissant l'équation différentielle, et ses dérivées partielles

def fx(vx, vy, t) :
    return W*vy

def fy(vx, vy, t) :
    return -W*vx

def dfx_dvx(vx,vy,t) :
    return 0

def dfx_dvy(vx,vy,t) :
    return W

def dfy_dvx(vx,vy,t) :
    return -W

def dfy_dvy(vx,vy,t) :
    return 0




m = 20 # nombre de points pour la variable indépendante
T = np.linspace(a,b,m) #liste des points de tests

H = 9 # nombre de noeuds de la couche cachée
epochs = 20000 #nombre d'itérations
alpha = 1e-5 #taux d'apprentissage pour la descente de gradients

model = Sequential()
model.add(Dense(H, activation='sigmoid', input_shape=(1,)))
model.add(Dense(2, activation='linear'))

grads = K.gradients(model.output, model.input)




