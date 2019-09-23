# -*- coding: utf-8 -*-
######################################################################
## A parte de las funciones aquí implementadas, necesitareis        ##
## np.random.uniform(rango[0], rango[1], size=(N, dim0, dim1, ...)) ## 
## que se corresponde con simula_unif(N, dim, rango)                ##              
######################################################################

import numpy as np
import matplotlib.pyplot as plt
import random as rd

np.random.seed(3)
# Función que devuelve un vector de forma <size> con valores aleatorios 
# extraídos de una normal de media <media> y varianza <sigma>.
# Se corresponde con simula_gaus(N, dim, sigma), en caso de llamarla
# con simula_gaus((N, dim0, dim1, ...), sigma).
def simula_gaus(size, sigma, media=None):
    media = 0 if media is None else media
    
    if len(size) >= 2:
        N = size[0]
        size_sub = size[1:]
        
        out = np.zeros(size, np.float64)
        
        for i in range(N):
            out[i] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=size_sub)
    
    else:
        out = np.random.normal(loc=media, scale=sigma, size=size)
    
    return out


# Función que devuelve los parámetros a y b de una recta aleatoria,
# y = a*x + b, tal que dicha recta corta al cuadrado definido por 
# por los puntos (intervalo[0], intervalo[0]) y 
# (intervalo[1], intervalo[1]).
def simula_recta(intervalo=(-1,1), ptos = None):
    if ptos is None: 
        m = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    
    a = (m[0,1]-m[1,1])/(m[0,0]-m[1,0]) # Calculo de la pendiente.
    b = m[0,1] - a*m[0,0]               # Calculo del termino independiente.
    
    return a, b

'''
    Transforma los parámetros de una recta 2d a los coeficientes de w.
    a: Pendiente de la recta.
    b: Término independiente de la recta.
'''
def line2coef(a, b):
    w = np.zeros(3, np.float64)
    #w[0] = a/(1-a-b)
    #w[2] = (b-b*w[0])/(b-1)
    #w[1] = 1 - w[0] - w[2]
    
    #Suponemos que w[1] = 1
    w[0] = -a
    w[1] = 1.0
    w[2] = -b
    
    return w

def coef2line(w):
    a = -w[0]/w[1]
    b = -w[2]/w[1]
    return a,b

'''
    Pinta los datos con su etiqueta y la recta definida por a y b.
    X: Datos (Intensidad promedio, Simetría).
    y: Etiquetas (-1, 1).
    a: Pendiente de la recta.
    b: Término independiente de la recta.
'''
def plot_datos_recta(X, y, a, b, title, xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    w = line2coef(a, b)
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = grid.dot(w)
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',
                      vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$w^tx$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white', label='Datos')
    ax.plot(grid[:, 0], a*grid[:, 0]+b, 'black', linewidth=2.0, label='Solucion')
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    ax.legend()
    plt.title(title)
    plt.show()
#####################################################################


######################################################################
##                                                                  ##         
##       EJERCICIO SOBRE LA COMPLEJIDAD DE H Y EL RUIDO             ##
##                                                                  ##
###################################################################### 
    np.random.seed(3)
    ###################################################################
    ##                      Ejercicio 1                              ##             
    ###################################################################    
    #   a) 

X = np.random.uniform(-50, 50,(50,2)) 
plt.figure('Ejercicio 1a')
plt.suptitle("Utilizando simula_unif()")
plt.scatter(X[:, 0], X[:, 1])
plt.show()

######################################################################
    #   b)

Y = simula_gaus((50,2), [5,7]) 
plt.figure("Ejercicio 1b")
plt.suptitle("Utilizando simula_gaus()")
plt.scatter(Y[:,0],Y[:,1])
plt.show()

    ###################################################################
    ##                      Ejercicio 2                              ##             
    ###################################################################
    #   a)

def distancia(a,b,x,y):
    return y - a*x - b

X = np.random.uniform(-50, 50,(100,2)) 
a,b = simula_recta() 

y = np.sign(distancia(a,b,X[:,0],X[:,1]))

plot_datos_recta(X,y,a,b, "")

######################################################################
    #   b)
# Función que genera ruido en las etiquetas calculando primero el numero
# de etiquetas a modificar y despues cambiando aleatoriamente las etiquetas
def meteRuido(y):
    y1 = y
    cont = 0
    size = 0
    #Calculamos el tamaño del vector y el 10%
    for i in y1:
        size+=1    
    p = size*0.1
    #Modificamos aleatoriamente las etiquetas
    for i in y1:
        if rd.randrange(10)%7 == 0:
            if p > 0:
                y1[cont] = -y1[cont]
                cont += 1
                p-=1 
    return y1
     
#Generamos las etiquetas con ruido
yr = y.copy()
yr = meteRuido(yr)
#Dibujamos la gráfica
plot_datos_recta(X,yr,a,b,"Ejercicio2b")

    ###################################################################
    ##                      Ejercicio 3                              ##             
    ###################################################################
#Definición de funciones
def funcion1(x,y):
    return (x-10)**2 + (y-20)**2 - 400

def funcion2(x,y):
    return (0.5*(x+10)**2 + (y-20)**2 - 400)

def funcion3(x,y):
    return 0.5*(x-10)**2 - (y+20)**2 - 400

def funcion4(x,y):
    return y - 20*(x**2) - 5*x + 3
#-----------------------------------------------------------------#
#Ponemos las funciones en función de x para la obtención de las gráficas
def f1(x):
    s1 = 20 - np.sqrt(-x**2+20*x+300)
    s2= np.sqrt(-x**2+20*x+300) + 20
    return s1,s2

def f2(x):
    s1 = 0.5*(40-np.sqrt(2)*np.sqrt(-x**2-20*x+700))
    s2 = 0.5*(np.sqrt(2)*np.sqrt(-x**2-20*x+700) + 40)
    return s1,s2

def f3(x):
    s1 = -((np.sqrt(x**2-20*x-700))/np.sqrt(2)) -20
    s2 = ((np.sqrt(x**2-20*x-700))/np.sqrt(2)) - 20 
    return s1,s2

def f4(x):
    return 20*x**2 + 5*x - 3
#-----------------------------------------------------------------#
#Establecemos el rango de valores
x = range(-50,51)

#Obtenemos las etiquetas 
y1 = np.sign(funcion1(X[:,0],X[:,1]))
y2 = np.sign(funcion2(X[:,0],X[:,1]))
y3 = np.sign(funcion3(X[:,0],X[:,1]))
y4 = np.sign(funcion4(X[:,0],X[:,1]))

#Dibujamos cada uno de los resultados

plt.figure("Ejercicio 3a")
plt.scatter(X[:,0],X[:,1],c=y1)
plt.plot(x, [f1(i) for i in x])
plt.axis([-50,50,-50,50])
plt.show()

plt.figure("Ejercicio 3b")
plt.scatter(X[:,0],X[:,1],c=y2)
plt.plot(x, [f2(i) for i in x])
plt.axis([-50,50,-50,50])
plt.show()

plt.figure("Ejercicio 3c")
plt.scatter(X[:,0],X[:,1],c=y3)
plt.plot(x, [f3(i) for i in x])
plt.axis([-50,50,-50,50])
plt.show()

plt.figure("Ejercicio 3d")
plt.scatter(X[:,0],X[:,1],c=y4)
plt.plot(x, [f4(i) for i in x])
plt.axis([-50,50,-50,50])
plt.show()

######################################################################
##                                                                  ##         
##                      MODELOS LINEALES                            ##
##                                                                  ##
######################################################################

    ###################################################################
    ##                      Ejercicio 1                              ##             
    ###################################################################
    
#   Función ajusta_PLA(datos, label, max_iter, vini) que calcula el 
#   hiperplano solución a un problema de clasificación binaria usando
#   el algoritmo PLA.
#   datos: matriz en la que cada item conn su etiqueta está representado
#          por una fila de la matriz
#   label: vector de etiquetas 
#   max_iter: número máximo de iteraciones permitidas
#   vini: valor inicial del vector
def ajusta_PLA(datos,label,max_iter,vini):
    w = vini
    it = 0
    continuar = 1
    while max_iter > it:
        if continuar:
            continuar = 0
            for i,x in enumerate(datos):
                if np.sign(np.dot(w,datos[i])) != label[i]:
                    w = w + label[i]*datos[i]
                    continuar = 1
        else:
            break
        it+=1
    return w,it

######################################################################
    #   a)

#   Función iteracionesMedias(n) que calcula el numero medio de iteraciones
#   que necesitan n ejecuciones de la función ajusta_PLA(...)
def iteracionesMedias(n,y):
    total = 0
    for i in range(n):
        x = np.random.uniform(0,1,3)
        w,it = ajusta_PLA(X,y,1000,x)
        total+=it
    
    return total/n
    
        
#   Ejecución del algoritmo PLA usando los datos del apartado 2a 
#   de la sección.1.
#   Vector de ceros
vini = np.zeros(3)
X = np.insert(X,2,1,axis=1)
w,it = ajusta_PLA(X,y,20,vini)
a,b = coef2line(w)
print("Iteraciones necesarias con etiquetas sin ruido")
print(it)
plot_datos_recta(X,y,a,b,"PLA 1")

# Ejecución del algoritmo 10 veces con vector aleatorio
media = iteracionesMedias(10,y)
print("La media de las iteraciones necesitadas es ")
print(media)
print("-------------------------------------")

######################################################################
    #   b)
#   Ejecución del algoritmo PLA usando los datos del apartado 2b 
#   de la sección.1.
vini = np.zeros(3)
w,it = ajusta_PLA(X,yr,1000,vini)
print("Iteraciones necesarias con etiquetas con ruido")
print(it)
plot_datos_recta(X,yr,a,b,"PLA 2")

# Ejecución del algoritmo 10 veces con vector aleatorio
media = iteracionesMedias(10,yr)
print("La media de las iteraciones necesitadas es ")
print(media)


