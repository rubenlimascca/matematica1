import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import libreria as bib

# Ajuste cuadratico
# define true model parameters
xa,xb=-3, 5 #intervalo
x = np.linspace(xa, xb, 100) # intervalo sobre el cual efectuamos el experimento
a, b, c, d= 1, 2, 5, 4
y_exact = a + b * x + c * x**2+ d*x**3

# simulate noisy data
m = 40
X = xa+(xb-xa) *np.random.rand(m) ### Genera numeros aleatorios entre -1,1
Y = a + b * X + c * X**2 +d*X**3+ 20*np.random.randn(m) # aquí variamos la dispersion

'''
# fit the data to the model using linear least square
A = np.vstack([X**0, X**1, X**2]) # see np.vander for alternative
sol, r, rank, sv = la.lstsq(A.T, Y)
'''


At = np.array([X**0, X**1, X**2,X**3])
auxMat = np.matmul(At,At.T)
np.reshape(Y,(m,1))
b = np.matmul(At,Y)
b=b.reshape(-1,1)
sol = bib.GaussElimWithPiv(auxMat,b)


y_fit=sol[0]+sol[1]*x+sol[2]*x**2+sol[3]*x**3
fig,ax=plt.subplots(figsize=(12,4))

ax.plot(X,Y,'go',alpha=0.5,label='Simulated data') # Grafica 
ax.plot(x,y_exact,'r',lw=2, label='True value $y=1+2x+3x^2+x^3$')  # Grafica 
ax.plot(x,y_fit,'b',lw=2,label='Least square fit')  # Grafica 
ax.set_xlabel(r"$x$",fontsize=18)
ax.set_ylabel(r"$y$",fontsize=18)
ax.legend(loc=2)
plt.show()

