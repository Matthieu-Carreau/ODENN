"""
Implements the resolution of an ODE using a neural network
"""

# tutoriel : https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough#train_the_model
# from custom losses : https://keras.io/api/losses/

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#physical paramters
lamb = 0.3
W=2*np.pi
p = (lamb, W)

x0 = 0
y0 = 0

def dy_dx(x,y, p) :
    """
    define the ODE : gives the derivate given the
    tensors x and y
    """
    lamb, W = p
    return lamb*y**2*W - lamb*W

def analytic(x, y0, p) :
    """
    Analytic solution to plot
    """
    #for Mz equation :

    lamb, W = p
    K = (y0-1) / (y0+1)
    Kexp = K*np.exp(2*lamb*W*x)
    Mz_ana = (1+Kexp) / (1-Kexp)
    return Mz_ana

    #return np.exp(x)

x_a = -2
x_b = 2
N = 50 #number of samples for the independant variable
training_points = np.linspace(x_a,x_b,N).reshape((N,1))
training_points = tf.convert_to_tensor(training_points, dtype=tf.float32)

load_model = False
save_model = True
load_filename = "models_for_Mz/sig_-2_2 min_loc2"
save_filename = "models_for_Mz/testMode"

learning_rate = 1e-4
epochs = 10000
testMode = True
display_step = min(max(50,epochs//100), 1000)

# Network Parameters
n_input = 1     # input layer number of neurons
n_hidden_1 = 16 # 1st layer number of neurons
n_hidden_2 = 8 # 2nd layer number of neurons
n_output = 1    # output layer number of neurons

initializers = ["zeros", "glorot_normal", "glorot_uniform","he_normal", "he_uniform", "random_normal", "random_uniform", "lecun_normal", "lecun_uniform"]
init = initializers[4]
#model definition :
model = tf.keras.Sequential([
  tf.keras.layers.Dense(n_hidden_1, activation=tf.nn.sigmoid, input_shape=(n_input,)), # input shape required
  tf.keras.layers.Dense(n_hidden_2, activation=tf.nn.sigmoid),
  tf.keras.layers.Dense(n_output)
])

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)


if load_model :
    model = tf.keras.models.load_model(load_filename)

model.summary()
#loss function
#https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/GradientTape

def loss_function(model, input_tensor, y0, p):


    if testMode :
        output = model(input_tensor)
        y = y0 + tf.math.multiply(input_tensor,output)
        target = analytic(input_tensor, y0, p)
        """
        print('IN :\n', input_tensor)
        print('OUT :\n',output)
        print('MUL :\n',y-y0)
        print((y-target)**2)
        """
        return tf.reduce_mean((y-target)**2)


    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        output = model(input_tensor)
        y = y0 + tf.math.multiply(input_tensor-x0,output)

    dy = tape.gradient(y, input_tensor)


    """
    print('IN :\n', tf.reshape(input_tensor, -1))
    print('OUT :\n',tf.reshape(output, -1))
    print('Y :\n',tf.reshape(y, -1))
    print('DY :\n',tf.reshape(y0, -1))

    print('IN :\n', input_tensor)
    print('OUT :\n',output)
    print('Y :\n',y)
    print('DY :\n',dy)

    """
    target = dy_dx(input_tensor, y, p) #lamb*W*(Mz**2-1)

    return tf.reduce_mean((dy-target)**2)







#gradient of loss

def grad(model, input_tensor, y0, p):
    with tf.GradientTape() as tape:
        loss_value = loss_function(model, input_tensor, y0, p)

    gradient = tape.gradient(loss_value, model.trainable_variables)

    return loss_value, gradient


losses = []
epochs_displayed = []
"""
print("BEFORE : \n",model(training_points))

#plot the estimation
nb_plotting_points = 200
plotting_points = np.linspace(-20,20,nb_plotting_points)
plotting_points = tf.convert_to_tensor(plotting_points, dtype=tf.float32)

#neural network estimation
output = model(plotting_points).numpy().reshape((nb_plotting_points))

plt.plot(plotting_points, output)
plt.show()
"""
output = model(training_points)
print("Extreme values of output after :",min(output), max(output))

for epoch in range(epochs) :
    loss_value, grads = grad(model, training_points, y0, p)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % display_step == 0 :
        print("Loss after",epoch,"/",epochs,"epochs :",loss_value.numpy())
        losses.append(loss_value.numpy())
        epochs_displayed.append(epoch)


loss_value, grads = grad(model, training_points, y0, p)
print("Final loss after",epochs,"epochs :",loss_value.numpy())
losses.append(loss_value.numpy())
epochs_displayed.append(epochs)


if save_model :
    model.save(save_filename)
"""
print("AFTER : \n",model(training_points))
#neural network estimation
output = model(plotting_points).numpy().reshape((nb_plotting_points))

plt.plot(plotting_points, output)
plt.show()
"""

"""
intermediate_0 = tf.keras.backend.function([model.layers[0].input],
                                  [model.layers[0].output])
layer_output = intermediate_0([training_points])
print(layer_output)

intermediate_1 = tf.keras.backend.function([model.layers[0].input],
                                  [model.layers[1].output])
layer_output = intermediate_1([training_points])
print(layer_output)

print(model.layers[1].weights)

print(model(training_points))
"""


if epochs>1:
    #plot the evolution of loss
    plt.plot(epochs_displayed, losses)
    plt.yscale('log')
    plt.show()


#plot the estimation
nb_plotting_points = 200
plotting_points = np.linspace(x_a,x_b,nb_plotting_points)
plotting_points = tf.convert_to_tensor(plotting_points, dtype=tf.float32)

#neural network estimation
output = model(plotting_points).numpy().reshape((nb_plotting_points))
y_NN = y0 + (plotting_points-x0)*output

print("Extreme values of output after :",min(output), max(output))
#analytic solution
y_ana = analytic(plotting_points, y0, p)

#training points
y_ana_training = analytic(training_points, y0, p)


plt.plot(plotting_points, y_NN, label='solution approchée')
plt.plot(plotting_points, y_ana, label='solution exacte')
plt.scatter(training_points, y_ana_training, label="points d'entraînement", color='red')

plt.legend()
plt.title('résolution par réseau de neurones')
plt.show()



