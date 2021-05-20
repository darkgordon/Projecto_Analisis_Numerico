#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
from sympy import*
from sympy.interactive import printing;
printing.init_printing(use_latex=true);
from IPython.display import display, Latex 

#test 1q13
# usar pandas y numpy

def nombre(3):
    contador=1;
<<<<<<< HEAD
# Correcto
# hayq eu  poner dash 
# dfdfggd
=======
>>>>>>> master


def Newton_algoritmo(F, J, x, eps):
    """
    Resuelve un sistema no linear Rn-Rn F(x)=0, ambos F y J deben ser funciones de x
    x es el valor de las coordenadas iniciales, y sigue hasta que ||F|| < eps que es una tolerancia
    """
    F_value = F(x)
    #display(Latex('$$ F(x) = '+ latex(simplify(F_value)) + '$$'))
    F_norm = np.linalg.norm(F_value, ord=2)  # l2 norm of vector
    contador_iteraciones = 0
    while abs(F_norm) > eps and   contador_iteraciones < 100:
        delta = np.linalg.solve(J(x), -F_value)
       # display(Latex('$$ F(x) = '+ latex(simplify(F_value)) + '$$'))
       # display(Latex('$$ J(x) = '+ latex(simplify(J(x))) + '$$'))
       # display(Latex('$$ SEL = '+ latex(simplify(delta)) + '$$'))
        x = x + delta
        display(Latex('$$ Iteracion = '+ latex(simplify(contador_iteraciones)) + '$$'))
        display(Latex('$$ SolucionSistema = '+ latex(simplify(x)) + '$$'))
        F_value = F(x) #test
        F_norm = np.linalg.norm(F_value, ord=2)
        contador_iteraciones += 1

    # Hasta que una solucion es encontrada o muchas iteraciones 
    if abs(F_norm) > eps:
        contador_iteraciones = -1
    return x, contador_iteraciones


def Sistema1():
    from numpy import cos, sin, pi, exp

    def F(x):
        eq1=3*x[0] - cos(x[1]*x[2]) -1/2
        eq2=x[0]**2 -81*(x[1]+0.1)**2 +sin(x[2])+1.06
        eq3= 1/exp(x[0]*x[1])+20*x[2]+(10*pi-3)/3
        return np.array(
            [eq1,eq2,eq3])

    def J(x):
        
        
        
        return np.array([[3,   x[2]*sin(x[1]*x[2]), x[1]*sin(x[1]*x[2])],
                         [  2*x[0],  -162*x[1] - 16.2,       cos(x[2])],
                         [-x[1]*exp(-x[0]*x[1]), -x[0]*exp(-x[0]*x[1]),20]])
#
    expected = np.array([0.5,0,-0.52359877])
    #esta es la tolerancia si quiere ser comprobada con algun esperado 
    tol = 1e-20
    x, n =  Newton_algoritmo(F, J, x=np.array([0.1,0.1,0.1]), eps=1e-10)
    error_norm = np.linalg.norm(expected - x, ord=2)
    if error_norm < tol:
        print('la norma del error es mas pequeña que la tolerancia')
        print('Numero maximo de iteraciones exedido')
        print ('norm of error =%g' % error_norm)
        print(' la tolerancia es', tol)
 


# In[48]:


Sistema1()


# In[36]:


test_Newton_system2()


# In[17]:


test_Newton_system3()


# In[ ]:





# In[ ]:




