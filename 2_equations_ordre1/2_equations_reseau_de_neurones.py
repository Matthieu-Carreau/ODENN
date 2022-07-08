"""
implémentation de résolution numérique approchée de l'équation 1 :
dy/dx = f(x,y), y(a) = A
à l'aide d'un réseau de neurones avec une couche cachée et des
fonctions d'activation sigmoid
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
"""
def fx(vx, vy, t) :
    return 0#W*vy

def fy(vx, vy, t) :
    return -cos(2*pi*t)#-W*vx

def dfx_dvx(vx,vy,t) :
    return 0

def dfx_dvy(vx,vy,t) :
    return 0#W

def dfy_dvx(vx,vy,t) :
    return 0#-W

def dfy_dvy(vx,vy,t) :
    return 0
"""
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


def save(P, filename) :
    w, b, ux, uy = P
    f = open(filename+".csv", 'w')
    for l in [w,b,ux,uy] :
        for i in range(len(l)) :
            f.write(str(l[i]))
            if i+1 != len(l) :
                f.write('; ')
        f.write('\n')
    f.close()

def load(filename) :
    f = open(filename+".csv")
    lines = f.readlines()
    f.close()
    floats = []
    for l in lines :
        valeurs = l.split(';')
        for v in valeurs :
            v = float(v)
        floats.append(np.array(valeurs, dtype='float64'))
    return floats[0],floats[1],floats[2],floats[3]

#sigmoid
def sig(x) :
    return 1/(1+np.exp(-x))

sig = np.vectorize(sig)


def N(t, w, b, ux, uy) :
    #Calcule la sortie du réseau de neurones
    z = t*w+b
    s = sig(z)
    Nx = np.dot(s,ux)
    Ny = np.dot(s,uy)
    return np.array([Nx, Ny])


def calcError(w, b, ux, uy) :
    #Calcule l'erreur
    E = 0
    for i in range(m) :

        s = sig(T[i]*w+b)

        Nx = np.dot(s,ux)
        Ny = np.dot(s,uy)

        vx = V_0x + (T[i]-a)*Nx
        vy = V_0y + (T[i]-a)*Ny

        ds = w*(s-s**2)

        ex = Nx + (T[i]-a)*np.dot(ux,ds) - fx(vx, vy, T[i])
        ey = Ny + (T[i]-a)*np.dot(uy,ds) - fy(vx, vy, T[i])

        """
        print("\ni=",i)
        print(vx,vx + (T[i]-a)*np.dot(ux,ds), fx(vx, vy, T[i]))
        print(vy,vy + (T[i]-a)*np.dot(uy,ds), fy(vx, vy, T[i]))
        print(ex**2 + ey**2)
        """

        E += ex**2 + ey**2

    return E


def calcGrad(w, b, ux, uy) :
    #calcule le gradient de l'erreur par rapport
    #aux 4 vecteurs représentant les paramètres
    grad_w = np.zeros(H)
    grad_b = np.zeros(H)
    grad_ux = np.zeros(H)
    grad_uy = np.zeros(H)

    for i in range(m):

        s = sig(T[i]*w+b)

        Nx = np.dot(s,ux)
        Ny = np.dot(s,uy)

        vx = V_0x + (T[i]-a)*Nx
        vy = V_0y + (T[i]-a)*Ny

        ds = w*(s-s**2)

        ex = Nx + (T[i]-a)*np.dot(ux,ds) - fx(vx, vy, T[i])
        ey = Ny + (T[i]-a)*np.dot(uy,ds) - fy(vx, vy, T[i])

        dfx_dx = dfx_dvx(vx,vy,T[i])
        dfx_dy = dfx_dvy(vx,vy,T[i])
        dfy_dx = dfy_dvx(vx,vy,T[i])
        dfy_dy = dfy_dvy(vx,vy,T[i])

        #w
        dex_dw = ux*(1+T[i]*w*(1-2*s)) - T[i]*(ux*dfx_dx + uy*dfx_dy)
        dex_dw = (s-s**2)*(ux*T[i]+(T[i]-a)*dex_dw)

        dey_dw = uy*(1+T[i]*w*(1-2*s)) - T[i]*(uy*dfy_dy + ux*dfy_dx)
        dey_dw = (s-s**2)*(uy*T[i]+(T[i]-a)*dey_dw)

        grad_w += 2*(ex*dex_dw + ey*dey_dw)

        #b
        dex_db = ux*(1+w) - (T[i]-a)*(ux*dfx_dx + uy*dfx_dy)
        dex_db = (s-s**2)*dex_db

        dey_db = uy*(1+w) - (T[i]-a)*(uy*dfy_dy + ux*dfy_dx)
        dey_db = (s-s**2)*dey_db

        grad_b += 2*(ex*dex_db + ey*dey_db)

        #ux
        dex_dux = 1 + w*(1-s) - (T[i]-a)*dfx_dx
        dex_dux = s*dex_dux

        dey_dux = -(T[i]-a)*s*dfy_dx

        grad_ux += 2*(ex*dex_dux + ey*dey_dux)

        #uy
        dey_duy = 1 + w*(1-s) - (T[i]-a)*dfy_dy
        dey_duy = s*dey_duy

        dex_duy = -(T[i]-a)*s*dfy_dx

        grad_uy += 2*(ex*dex_duy + ey*dey_duy)


    return grad_w, grad_b, grad_ux, grad_uy




m = 20 # nombre de points pour la variable indépendante
T = np.linspace(a,b,m) #liste des points de tests

H = 9 # nombre de noeuds de la couche cachée
epochs = 20000 #nombre d'itérations
alpha = 1e-5 #taux d'apprentissage pour la descente de gradients

#initialisation des paramètres
#poids entre l'entrée et la couche cachée
w = np.random.randn((H))
#biais de la couche cachée
b = np.random.randn((H))
#poids entre la couche cachée et la sortie x
ux = np.random.randn((H))
#poids entre la couche cachée et la sortie y
uy = np.random.randn((H))

w,b,ux,uy = load("test H = 9, E = 45")
#save((w,b,ux, uy), 'test24')

"""
w = np.array([0, -8, -4, -4, 3,0.01,0.03,-0.01,0.004], dtype='float64')
b = np.array([0, 4, 1, 4, -2,0.01,0.03,-0.01,0.004], dtype='float64')
ux = np.array([0, -8, 0,0,0,0.01,0.03,-0.01,0.004], dtype='float64')
uy = np.array([0, 0, -14*1.3, 5*1.4, -2.5*1.3,0.01,0.03,-0.01,0.004], dtype='float64')
"""
erreur  = calcError(w, b, ux, uy)
meilleurs_P = (w+0, b+0, ux+0, uy+0)
meilleure_E = erreur
#entrainement :
for k in range(epochs) :
    if (k%500 == 0) :
        erreur  = calcError(w, b, ux, uy)
        print("Erreur après",k,"itérations :", erreur)
        if erreur <meilleure_E :
            meilleure_E = erreur
            meilleurs_P = (w+0, b+0, ux+0, uy+0)


    gw,gb,gux, guy = calcGrad(w, b, ux, uy)
    w -= alpha*gw
    b -= alpha*gb
    ux -= alpha*gux
    uy -= alpha*guy

print()

erreur  = calcError(w, b, ux, uy)
print("Erreur finale après",epochs,"itérations :", erreur)

if erreur <meilleure_E :
    meilleure_E = erreur
    meilleurs_P = (w+0, b+0, ux+0, uy+0)

save(meilleurs_P, 'test H = '+str(H)+', E = '+str(int(meilleure_E)))


#solution trouvée
V_NN = np.array([V_0 + (T[i]-a)*N(T[i], w, b, ux, uy) for i in range(m)])
#solution analytique
V_ana = np.array([[V_0x*cos(W*T[i])+V_0y*sin(W*T[i]), -V_0x*sin(W*T[i])+V_0y*cos(W*T[i])] for i in range(m)])
#V_ana = np.array([[0,1-sin(W*T[i])/2/pi] for i in range(m)])

#affichage 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs=V_NN[:,0], ys=V_NN[:,1], zs=T, label='solution approchée')
ax.plot(xs=V_ana[:,0], ys=V_ana[:,1], zs=T, label='solution analytique')
plt.legend()
plt.title('résolution par réseau de neurones')
plt.show()
