"""
implémentation de résolution numérique approchée de l'équation 1 :
dy/dx = f(x,y), y(a) = A
à l'aide de la la méthode de descente de gradients sur la fonction d'erreur
"""


import numpy as np

#nombre de coefficients de Fourier ajustables
M = 10
M_range = np.arange(M) + 1

# nombre de points pour la variable indépendante
N = 100
X = np.linspace(0,1,N)


def calcGrad(A) :
    #Calcule le gradient de l'erreur par rapport au vecteur
    #des coefficients
    grad = np.zeros((M))
    V = np.cos(2*np.pi*X)
    for m in range(1, M+1) :
        for i in range(N) :
            V[i] += 2*np.pi*m*A[m-1]*np.cos(2*np.pi*m*X[i])
    for l in range(1, M+1) :
        W = np.pi*l*np.cos(2*np.pi*l*X)
        grad[l-1] = np.dot(V,W)
    return grad



alpha = 1e-5 #taux d'apprentissage pour la descente de gradients
epochs = 1000 #nombre d'itération

#initialisation des coefficients
A = np.random.randn((M))

#descente de gradients
for k in range(epochs) :
    if k%100==0 :
        print("Coefficients après",k,"itérations :",A)
    A -= alpha*calcGrad(A)

print()
print("Coefficients finaux après",epochs,"itérations :",A)
