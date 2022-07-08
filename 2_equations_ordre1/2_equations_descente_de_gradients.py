"""
implémentation de résolution numérique approchée des 2 éuqations couplés du problème de précession dans un champ constant :
dv_x/dt = Wv_y
dv_y/dt = -Wv_x
avec v_x(0) = V_0, v_y(0) = 0
à l'aide de la la méthode de descente de gradients sur la fonction d'erreur
"""

import numpy as np

#nombre de coefficients de Fourier ajustables
M = 10
M_range = np.arange(M) + 1

# nombre de points pour la variable indépendante
N = 100
T = np.linspace(0,1,N)

#paramètres du problème
W = 2*np.pi #pulsation
V0 = 1 #vitesse initiale

#pour éviter la répétition inutile de calculs :
#matrice de coefficients (m*W*T[i]) de taille NxM
#avec première coordonnée i, et deuxème m
mat = W*np.matmul(T.reshape((N,1)), M_range.reshape((1,M)))
cos = np.cos(mat)
sin = np.sin(mat)

def calcGrad(A,B) :
    #calcule le gradient de l'erreur par rapport aux
    #2 vecteurs de paramètres A et B
    grad_A = np.zeros((M))
    grad_B = np.zeros((M))

    for i in range(N) :
        #valeurs de vx et vy à l'instant T[i]
        vx = V0 + np.dot(A, cos[i]-1) + np.dot(B, sin[i])
        vy = - np.dot(M_range*A, sin[i]) + np.dot(M_range*B, cos[i])

        #valeurs des dérivées de vx et vy à l'instant T[i]
        dvx = W*(-np.dot(M_range*A, sin[i]) + np.dot(M_range*B, cos[i]))
        dvy = W*(-np.dot(M_range*A, cos[i]) - np.dot(M_range*B, sin[i]))

        #différences
        ex = dvx - W*vy
        ey = dvy + W*vx

        #on incrémente les gradients
        #pour A
        dex_dA = W*(1-M_range)*sin[i]
        dey_dA = W*((1-M_range)*cos[i]-1)
        grad_A += ex*dex_dA + ey*dey_dA
        #pour B
        dex_dB = W*(M_range-1)*cos[i]
        dey_dB = W*(1-M_range)*sin[i]
        grad_B += ex*dex_dB + ey*dey_dB

    return grad_A, grad_B



alpha = 1e-5 #taux d'apprentissage pour la descente de gradients
epochs = 10000 #nombre d'itération

#initialisation des coefficients
A = np.random.randn((M))
B = np.random.randn((M))

#descente de gradients
for k in range(epochs) :
    if k%500==0 :
        print("\nCoefficients après",k,"itérations :\n",A,"\n", B)

    grad_A, grad_B = calcGrad(A,B)
    A -= alpha*grad_A
    B -= alpha*grad_B


print()
print("Coefficients finaux après",epochs,"itérations :",A,B)
