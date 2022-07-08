"""
implémentation de résolution numérique approchée de l'équation 1 :
dy/dx = f(x,y), y(a) = A
à l'aide d'un réseau de neurones avec une couche cachée et des
fonctions d'activation sigmoid
"""

import numpy as np
import matplotlib.pyplot as plt

cos = np.cos
pi = np.pi
sin = np.sin

# données du problème
#borne de l'intervalle
a = 0
b = 1

#condition initiale
A = 1

def f(x,y) :
    return -cos(2*pi*x)

def df_dy(x,y) :
    """renvoie df/dy (x,y)"""
    return 0


def save(P, filename) :
    w, b, v = P
    f = open(filename+".csv", 'w')
    for l in [w,b,v] :
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
    return floats[0],floats[1],floats[2]

#sigmoid
def sig(x) :
    return 1/(1+np.exp(-x))

sig = np.vectorize(sig)



def N(x, w, b, v) :
    #Calcule la sortie du réseau de neurones
    z = x*w+b
    s = sig(z)
    return np.dot(s,v)


def calcError(w, b, v) :
    #Calcule l'erreur
    E = 0
    for i in range(m) :

        s = sig(X[i]*w+b)
        e = np.dot(s,v)
        e += (X[i]-a)*np.sum(v*w*(s-s**2))
        e -= (f(X[i], np.dot(s,v)))
        E += e**2
    return E


def calcGrad(w, b, v) :
    #calcule le gradient de l'erreur par rapport
    #aux 3 vecteurs représentant les paramètres
    grad_w = np.zeros(H)
    grad_b = np.zeros(H)
    grad_v = np.zeros(H)

    for i in range(m):

        s = sig(X[i]*w+b)
        df = df_dy(X[i], np.dot(s,v))

        e = np.dot(s,v)
        e += (X[i]-a)*np.sum(v*w*(s-s**2))
        e -= (f(X[i], np.dot(s,v)))

        #w
        de_dw = 1 + (X[i]-a)*(w*(1-2*s)-df)
        de_dw = X[i]*(s-s**2)*v*de_dw

        grad_w += 2*e*de_dw

        #b
        de_db = 1 + (X[i]-a)*(w*(1-2*s)-df)
        de_db = (s-s**2)*v*de_db

        grad_b += 2*e*de_db

        #v
        de_dv = 1 + (X[i]-a)*(w*(1-s)-df)
        de_dv = s*de_dv

        grad_v += 2*e*de_dv


    return grad_w, grad_b, grad_v




m = 20 # nombre de points pour la variable indépendante
X = np.linspace(a,b,m) #liste des points de tests

H = 4 # nombre de noeuds de la couche cachée
epochs = 0 #nombre d'itérations
alpha = 1e-2 #taux d'apprentissage pour la descente de gradients

#initialisation des paramètres
#poids entre l'entrée et la couche cachée
w = np.random.randn((H))
#biais de la couche cachée
b = np.random.randn((H))
#poids entre la couche cachée et la sortie
v = np.random.randn((H))


#w,b,v = load("H = 4, E = 0")

#entrainement :
for k in range(epochs) :
    if (k%500 == 0) :
        print("Erreur après",k,"itérations :", calcError(w, b, v))
    gw,gb,gv = calcGrad(w, b, v)
    w -= alpha*gw
    b -= alpha*gb
    v -= alpha*gv

print()
print("Erreur finale après",epochs,"itérations",calcError(w, b, v))
print("paramètres trouvés :",w,b,v)
save((w,b,v), 'H = '+str(H)+', E = '+str(int(calcError(w, b, v))))

#solution trouvée
Y = [A + (X[i]-a)*N(X[i], w, b, v) for i in range(m)]
#solution analytique
Z = [1 - sin(2*pi*X[i])/2/pi for i in range(m)]


plt.plot(X,Z, label='solution analytique')
plt.plot(X,Y, label='solution approchée')
plt.legend()
plt.title('Solution approchée et analytique')
plt.show()
