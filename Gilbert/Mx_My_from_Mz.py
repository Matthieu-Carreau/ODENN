"""
Implements the resolution of the Mx and My components for the equation of precession with gilbert term, given a model for the evolution of Mz component
"""

# tutoriel : https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough#train_the_model
# from custom losses : https://keras.io/api/losses/

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#physical paramters
alpha = 1
lamb = 0.3/alpha
W=2*np.pi*alpha
p = (lamb, W)

t0 = 0
M0 = tf.constant([1,0], dtype = 'float32')



def dM_dt(T,Mx, My, Mz, p) :
    """
    define the ODE : gives the derivate given the
    tensors x and y
    """
    lamb, W = p

    dMx = W*(My + lamb*tf.multiply(Mx,Mz))
    dMy = W*(-Mx + lamb*tf.multiply(My,Mz))

    #dM = tf.concat((dMx, dMy), axis=1)
    return dMx, dMy


def analytic(T, M0, Mz_0, p) :
    """
    Analytic solution to plot
    """
    lamb, W = p

    K = (Mz_0-1) / (Mz_0+1)
    exp_vector = tf.exp(lamb*W*T)
    F = exp_vector*(K-1)/(K*exp_vector**2-1)

    Mx =tf.multiply( F,(  tf.cos(W*T)*M0[0] + tf.sin(W*T)*M0[1]))
    My = tf.multiply(F,(- tf.sin(W*T)*M0[0] + tf.cos(W*T)*M0[1]))
    M = tf.concat((Mx, My), axis=1)
    return M

    #return np.exp(x)

t_a = 0
t_b = 1
N = 10 #number of samples for the independant variable
training_points = np.linspace(t_a,t_b,N).reshape((N,1))
training_points = tf.convert_to_tensor(training_points, dtype=tf.float32)

load_model = False
save_model = False
load_filename = "models_for_Mx_My/relu testMode"
save_filename = "models_for_Mx_My/relu pretrained then without"

#train_with_Mz_model

learning_rate = 2e-4
epochs = 0
testMode = True
display_step = min(max(50,epochs//100), 1000)



#analytic version of Mz :
Mz_0 = 0
K = (Mz_0-1) / (Mz_0+1)
Kexp = K*tf.exp(2*lamb*W*training_points)
Mz_ana = (1+Kexp) / (1-Kexp)

#NN version of Mz :

Mz_model_filename = "models_for_Mz/h1= 32 l = 0.3"
Mz_model = tf.keras.models.load_model(Mz_model_filename)
Mz_NN = Mz_model(training_points)


#analytic result for M vector
M_analytic = analytic(training_points, M0, Mz_0, p)


# Network Parameters
n_input = 1     # input layer number of neurons
n_hidden_1 = 32 # 1st layer number of neurons
n_hidden_2 = 16 # 2nd layer number of neurons
n_output = 2   # output layer number of neurons

initializers = ["zeros", "glorot_normal", "glorot_uniform","he_normal", "he_uniform", "random_normal", "random_uniform", "lecun_normal", "lecun_uniform"]
init = initializers[4]

#model definition :
model = tf.keras.Sequential([
  tf.keras.layers.Dense(n_hidden_1, activation=tf.nn.sigmoid, input_shape=(n_input,)),#kernel_initializer=init,
    #bias_initializer="zeros"),  # input shape required
  tf.keras.layers.Dense(n_hidden_2, activation=tf.nn.sigmoid),#kernel_initializer="he_uniform",
    #bias_initializer="zeros"),
  tf.keras.layers.Dense(n_output)#kernel_initializer=init, bias_initializer="zeros")
])

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)


if load_model :
    model = tf.keras.models.load_model(load_filename)


#loss function
#https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/GradientTape

def loss_function(model, input_tensor, M0, Mz, p):


    if testMode :
        output = model(input_tensor)
        M = M0 + tf.math.multiply(input_tensor,output)
        target = M_analytic
        """
        print('IN :\n', input_tensor)
        print('OUT :\n',output)
        print('MUL :\n',M-M0)
        print((M-target)**2)
        print(tf.reduce_mean((M-target)**2))
        """
        return tf.reduce_mean((M-target)**2)


    with tf.GradientTape(persistent = True) as tape:
        tape.watch(input_tensor)
        output = model(input_tensor)
        """
        print('T :\n', input_tensor-t0)
        print('OUT :\n',output)
        print('MUL :\n',tf.math.multiply(input_tensor-t0,output))
        """
        M = M0 + tf.math.multiply(input_tensor-t0,output)
        Mx = M[:,0]
        My = M[:,1]


    dMx = tape.gradient(Mx, input_tensor)
    dMy = tape.gradient(My, input_tensor)

    """
    #dM = tape.gradient(M, input_tensor)

    dM = tape.batch_jacobian(M, input_tensor)
    dM = tf.reshape(dM, (N,2))
    """

    """
    print('IN :\n', tf.reshape(input_tensor, -1))
    print('OUT :\n',tf.reshape(output, -1))
    print('Y :\n',tf.reshape(y, -1))
    print('DY :\n',tf.reshape(y0, -1))
    """
    print('IN :\n', input_tensor)
    print('OUT :\n',output)
    print('M :\n',M)
    print('DM :\n',dMx, dMy)


    dMx_target, dMy_target = dM_dt(input_tensor, Mx, My, Mz, p) #lamb*W*(Mz**2-1)
    """
    print('M :\n',M)
    print('DM :\n',dM)
    print('DM :\n',dM)
    print('target :\n',target)
    """
    Ex = tf.reduce_mean((dMx-dMx_target)**2)
    Ey = tf.reduce_mean((dMy-dMy_target)**2)

    return Ex + Ey




#gradient of loss

def grad(model, input_tensor, M0, Mz, p):
    with tf.GradientTape() as tape:
        loss_value = loss_function(model, input_tensor, M0, Mz, p)

    gradient = tape.gradient(loss_value, model.trainable_variables)

    return loss_value, gradient


losses = []
epochs_displayed = []
for epoch in range(epochs) :
    loss_value, grads = grad(model, training_points, M0, Mz_NN, p)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % display_step == 0 :
        print("Loss after",epoch,"/",epochs,"epochs :",loss_value.numpy())
        losses.append(loss_value.numpy())
        epochs_displayed.append(epoch)

loss_value, grads = grad(model, training_points, M0, Mz_NN, p)
print("Final loss after",epochs,"epochs :",loss_value.numpy())
losses.append(loss_value.numpy())
epochs_displayed.append(epochs)


if save_model :
    model.save(save_filename)

if epochs > 1 :
    #plot the evolution of loss
    plt.plot(epochs_displayed, losses)
    plt.yscale('log')
    plt.show()

#plot the estimation
nb_plotting_points = 200
plotting_points = np.linspace(t_a-(t_b-t_a)*0.3,t_b+(t_b-t_a)*0.3,nb_plotting_points)
plotting_points_tensor = tf.convert_to_tensor(plotting_points.reshape((nb_plotting_points,1)), dtype=tf.float32)

#neural network estimation
output = model(plotting_points_tensor)
Mx_NN, My_NN = M0[0]+plotting_points*output[:,0], M0[1]+plotting_points*output[:,1]


#analytic solution
M_ana = analytic(plotting_points_tensor, M0, Mz_0, p)
Mx_ana =  M_ana[:,0]
My_ana = M_ana[:,1]

#training points
Mx_ana_training_points =  M_analytic[:,0]
My_ana_training_points = M_analytic[:,1]


#3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(        xs=Mx_NN,
                ys=My_NN,
                zs=plotting_points,
                label='solution approchée')

ax.plot(        xs=Mx_ana,
                ys=My_ana,
                zs=plotting_points,
                label='solution exacte')



ax.scatter(     xs=Mx_ana_training_points,
                ys=My_ana_training_points,
                zs=tf.reshape(training_points, (N)),
                color='red')

plt.legend()
plt.title('résolution par réseau de neurones')
plt.show()



#plot neural network output
output = model(plotting_points).numpy()#.reshape((nb_plotting_points,2))


plt.plot(plotting_points, output[:,0], label = 'output x')
plt.plot(plotting_points, output[:,1], label = 'output y')
plt.legend()
plt.show()
