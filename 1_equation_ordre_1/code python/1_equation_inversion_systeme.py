"""
implémentation de résolution numérique approchée de l'équation 1 :
dy/dx = f(x,y), y(a) = A
à l'aide de la résolution du système linéaire correspondant à l'annulation des dérivées partielles de la fonction d'erreur
"""

import numpy as np

#nombre de coefficients de Fourier ajustables
M = 50
# nombre de points pour la variable indépendante
N = 100
#liste des points de test
X = np.linspace(0,1,N)

#Matrice M représentant le système
mat = np.zeros((M,M))

for l in range(1,M+1) :
    for m in range(1,M+1) :
        mat[m-1,l-1] = m*l*np.dot(np.cos(2*np.pi*m*X), np.cos(2*np.pi*l*X))
mat *= 2*np.pi

#matrice inverse
mat_inv = np.linalg.inv(mat)

#Vecteur b
b = np.zeros(M)
for l in range(1,M+1) :
    b[l-1] = -l*np.dot(np.cos(2*np.pi*X), np.cos(2*np.pi*l*X))

#Solution
A = np.matmul(mat_inv,b)

print("coefficients trouvés :",A)

