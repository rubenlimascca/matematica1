import numpy as np
import libreria as lib
import matplotlib.pyplot as plt
import numpy.polynomial as P

'''Interpolación'''
x=np.array([2,3,4,5,7,9,10,1]) #Cordenadas "x" de los puntos a interpolar
y=np.array([1,3,-4,-5,17,-19,-20,5]) # Coordenadas "y" de los puntos a interpolar
y=np.sin(x)

pol=lib.interpLagrange(x,y)

print(pol)
print(pol(x))

"Gráfica"
a=x.min()
b=x.max()
xx=np.linspace(a,b,200)
yy=pol(xx)
yy_exacta=np.sin(xx)

fig, ax=plt.subplots(figsize=(10,8))
ax.plot(xx,yy,'b',lw=2,label='Polinomio interpolante')
ax.plot(x,y,'ro',alpha=0.6,label='Datos')
ax.plot(xx,yy_exacta,'g',lw=2,label='solución exacta')
ax.legend(loc=2)
ax.set_xlabel(r"$x$",fontsize=10)
ax.set_ylabel(r"$y$",fontsize=10)
plt.grid()
plt.show()
