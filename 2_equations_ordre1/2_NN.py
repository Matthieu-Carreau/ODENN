# tutoriel : https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough#train_the_model
# from custom losses : https://keras.io/api/losses/

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#physical paramters
V_0 = tf.constant([1,0], dtype='float32')
W = 2*np.pi
#matrix definig the differential equation
P = W*np.array([[0,1], [-1, 0]], dtype = 'float32')

N = 100 #number of samples for the independant variable
training_points = np.linspace(-1,1,N)
training_points = tf.convert_to_tensor(training_points, dtype=tf.float32)

load_model = False
load_filename = "models/2_NN_direct_training_N=10"
save_model = True
save_filename = "models/2_NN_direct_training_N=100"
learning_rate = 0.007
epochs = 50000
display_step = min(max(1,epochs//100), 1000)

# Network Parameters
n_input = 1     # input layer number of neurons
n_hidden_1 = 32 # 1st layer number of neurons
n_hidden_2 = 8 # 2nd layer number of neurons
n_output = 2    # output layer number of neurons



#model definition :
model = tf.keras.Sequential([
  tf.keras.layers.Dense(n_hidden_1, activation=tf.nn.sigmoid, input_shape=(n_input,)),  # input shape required
  tf.keras.layers.Dense(n_hidden_2, activation=tf.nn.sigmoid),
  tf.keras.layers.Dense(n_output)
])

if load_model :
    model = tf.keras.models.load_model(load_filename)


#loss function
#https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/GradientTape

def loss_function(model, input_tensor, V_0):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_tensor)
        output = model(input_tensor)
        vx, vy = V_0[0]+input_tensor*output[:,0], V_0[1]+input_tensor*output[:,1]

    dvx = tape.gradient(vx, input_tensor)
    dvy = tape.gradient(vy, input_tensor)

    ex = dvx - W*vy
    ey = dvy + W*vx

    return tf.reduce_mean(ex**2 + ey**2)



#gradient of loss

def grad(model, input_tensor, V_0):
    with tf.GradientTape() as tape:
        loss_value = loss_function(model, input_tensor, V_0)

    gradient = tape.gradient(loss_value, model.trainable_variables)

    return loss_value, gradient



optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

print("BEFORE : \n",model(training_points))

#plot the estimation
nb_plotting_points = 200
plotting_points = np.linspace(-20,20,nb_plotting_points)
plotting_points = tf.convert_to_tensor(plotting_points, dtype=tf.float32)

#neural network estimation
output = model(plotting_points).numpy()

plt.plot(plotting_points, output[:,0])
plt.plot(plotting_points, output[:,1])
plt.show()

losses = []
epochs_displayed = []

for epoch in range(epochs) :
    loss_value, grads = grad(model, training_points, V_0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % display_step == 0 :
        print("Loss after",epoch,"/",epochs,"epochs :",loss_value.numpy())
        losses.append(loss_value.numpy())
        epochs_displayed.append(epoch)

loss_value, grads = grad(model, training_points, V_0)
print("Final loss after",epochs,"epochs :",loss_value.numpy())
losses.append(loss_value.numpy())
epochs_displayed.append(epochs)

print("AFTER : \n",model(training_points))

#plot the estimation
nb_plotting_points = 200
plotting_points = np.linspace(-20,20,nb_plotting_points)
plotting_points = tf.convert_to_tensor(plotting_points, dtype=tf.float32)

#plot neural network output
output = model(plotting_points).numpy()#.reshape((nb_plotting_points,2))


plt.plot(plotting_points, output[:,0])
plt.plot(plotting_points, output[:,1])
plt.show()

if save_model :
    model.save(save_filename)



#plot the evolution of loss
plt.plot(epochs_displayed, losses)
plt.yscale('log')
plt.show()

#plot the estimation
plotting_points = np.linspace(-1,1,200)
plotting_points = tf.convert_to_tensor(plotting_points, dtype=tf.float32)

#neural network estimation
output = model(plotting_points)
vx_NN, vy_NN = V_0[0]+plotting_points*output[:,0], V_0[1]+plotting_points*output[:,1]

#analytic solution
vx_ana =  V_0[0]*np.cos(W*plotting_points) + V_0[1]*np.sin(W*plotting_points)
vy_ana = -V_0[0]*np.sin(W*plotting_points) + V_0[1]*np.cos(W*plotting_points)

#training points
vx_ana_training_points =  V_0[0]*np.cos(W*training_points) + V_0[1]*np.sin(W*training_points)
vy_ana_training_points = -V_0[0]*np.sin(W*training_points) + V_0[1]*np.cos(W*training_points)


#3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(        xs=vx_NN,
                ys=vy_NN,
                zs=plotting_points,
                label='solution approchée')

ax.plot(        xs=vx_ana,
                ys=vy_ana,
                zs=plotting_points,
                label='solution exacte')



ax.scatter(     xs=vx_ana_training_points,
                ys=vy_ana_training_points,
                zs=training_points,
                color='red')

plt.legend()
plt.title('résolution par réseau de neurones')
plt.show()















