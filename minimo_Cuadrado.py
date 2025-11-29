import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import libreria as bib

# define true model parameters
x = np.linspace(-1, 1, 100) # intervalo sobre el cual efectuamos el experimento
a, b, c = 1, 2, 150
y_exact = a + b * x + c * x**2

# simulate noisy data
m = 200
X = 1 - 2 * np.random.rand(m) ### Genera numeros aleatorios entre -1,1
Y = a + b * X + c * X**2 + 2*np.random.randn(m)

# fit the data to the model using linear least square
A = np.vstack([X**0, X**1, X**2]) # see np.vander for alternative
sol, r, rank, sv = la.lstsq(A.T, Y)

'''
21 At = np.array([X**0, X**1, X**2])
22 auxMat = np.matmul(At,At.T)
23 np.reshape(Y,(m,1))
24 b = np.matmul(At,Y)
25 sol = bib.GaussElimPiv(auxMat,b)

'''
y_fit=sol[0]+sol[1]*x+sol[2]*x**2
fig,ax=plt.subplots(figsize=(12,4))

ax.plot(X,Y,'go',alpha=0.5,label='Simulated data') # Grafica 
ax.plot(x,y_exact,'r',lw=2, label='True value $y=1+2x+3x^2$')  # Grafica 
ax.plot(x,y_fit,'b',lw=2,label='Least square fit')  # Grafica 
ax.set_xlabel(r"$x$",fontsize=18)
ax.set_ylabel(r"$y$",fontsize=18)
ax.legend(loc=2)
plt.show()

