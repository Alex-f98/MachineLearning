import streamlit as st


st.markdown("""
Indice:

* [Python](#python).
* [Imports](#imports).
* [Listas](#listas).
* [Tuplas](#tuplas).
* [Diccionarios y sets](#diccionarios-y-sets).
---
""", unsafe_allow_html=True)

st.markdown("### Python")
st.markdown("""
Python es un lenguaje de programación fácil de aprender y potente. Tiene **eficientes** estructuras de datos de alto nivel y un enfoque sencillo pero eficaz para **Programación** orientada a objetos. La **elegante** sintaxis y **tipificación** dinámica de Python, junto con su naturaleza interpretada, lo convierte en un lenguaje ideal para el scripting y desarrollo rápido de aplicaciones en muchas áreas y en la mayoría de las plataformas.

El **intérprete** de Python y la extensa biblioteca estándar están disponibles gratuitamente en forma de código fuente o binario para todas las plataformas principales desde la web de Python, https://www.python.org/, y puede distribuirse libremente. El mismo sitio también contiene distribuciones y **punteros** a muchos módulos Python gratuitos de terceros, programas y herramientas, y documentación adicional.

El **intérprete** de Python se **amplía** fácilmente con nuevas funciones y tipos de datos implementado en C o C++ (u otros lenguajes **llamables** desde C). Python también lo es adecuado como lenguaje de **extensión** para aplicaciones personalizables.

**Características**

- **Interpretado**: Python es interpretado, lo que significa que se ejecuta línea por línea sin necesidad de compilación previa. Esto permite una rápida **iteración** y **depuración**.

- **Dinámico**: Python es **dinámicamente** tipado, lo que permite declarar variables sin especificar su tipo. El tipo se determina en tiempo de **ejecución**.

- **Orientado a Objetos**: Python soporta **programación** orientada a objetos (OOP), **permitiendo** la creación de clases y objetos.

""")

st.markdown("""### imports""")
st.markdown("""
**¿Qué es la declaración `import`?**

La instrucción `import` permite **usar** código definido en otros **módulos** o paquetes dentro de tu programa. En esencia:
1. **Localiza y carga** el módulo solicitado (**ejecutando** su código si es necesario).
2. **Vincula** nombres en tu espacio de trabajo local para poder acceder a las funciones, clases o variables de ese módulo.


#### Formas principales de uso
- **Importación básica**
  ```python
  import math
  ```
  - Carga el módulo completo.
  - Se accede con su nombre: `math.sqrt(16)`.

- **Importación con alias**
  ```python
  import numpy as np
  ```
  - Permite usar un nombre más corto o conveniente.

- **Importación de elementos específicos**
  ```python
  from math import sqrt, pi
  ```
  - Trae solo los nombres indicados.
  - Se accede directamente: `sqrt(16)`.

- **Importación con alias de elementos**
  ```python
  from math import sqrt as raiz
  ```
  - El nombre local será `raiz`.

- **Importación relativa (en paquetes)**
  ```python
  from . import modulo_local
  from ..subpkg import herramienta
  ```
  - Útil para organizar proyectos grandes con varios submódulos.

- **Importación comodín**
  ```python
  from math import *
  ```
  - Importa todos los nombres públicos del módulo.
  - ⚠️ No se recomienda porque puede generar conflictos de nombres y falta de claridad.


""")

st.markdown("""### Listas""")
st.markdown("""
    Las listas son una estructura de datos. 
Usaremos listas para poder modelar datos compuestos, pero cuya cantidad y valor varían a lo largo del tiempo. 
Son secuencias mutables y vienen dotadas de una variedad de operaciones muy útiles.
La notación para lista es una secuencia de valores encerrados entre corchetes y separados por
comas.

**Características:**

* Arreglos dinámicos.
* Mutables.
* La suma de listas es equivalente a concatenar listas.  (❗Nota: *en numpy son arrays y se suman m.a.m*)
* Admite elementos de distintos tipos.
* Para añadir elementos a una lista usaremos corchetes []

ej 1:
```python
lista1 = [343, 35645, 4.5, "dfghjyrt", foo]
print(lista1)
#[343, 35645, 4.5, 'dfghjyrt', <function foo at 0x7949b4d20e00>]
```
Nota que en el ejemplo en la lista se guardaron valores enteros, flotantes, strings y una función.

ej 2:
```python
l1 = [1, 2., 3, 4, 5]
l2 = [6, 7, 8, 9, 10]
l3 = l1 + l2              #creo una nueva lista que contiene la concatenacion de l1 y l2

print( "len(l1):            ", len(l1) )
print( "len(l2):            ", len(l2) )
print( "len(l3 = l1 + l2):  ", len(l3) )

print("l3:", l3)

#Output: 
#len(l1)            :  5
#len(l2)            :  5
#len(l3 = l1 + l2)  :  10
#l3: [1, 2.0, 3, 4, 5, 6, 7, 8, 9, 10]
```
Nota la "suma" de dos *listas* resulta en una nueva lista con los valores de ambas.

ej 3: *siendo N el largo de la lista, se indexa desde 0 hasta N-1*
```python
for indice, valor in enumerate(l1):
  print(f"l1[{indice}] =  {valor},    type: {type(valor)}")
    
#l1[0] =  1,    type: <class 'int'>
#l1[1] =  2.0,  type: <class 'float'>
#l1[2] =  3,    type: <class 'int'>
#l1[3] =  4,    type: <class 'int'>
#l1[4] =  5,    type: <class 'int'>    
```

Nota: `enumerate()` es una función que devuelve un iterador que produce tuplas de (índice, valor) para cada elemento de la lista.
""")


st.markdown("""### Tuplas""")
st.markdown("""
Una tupla es una secuencia de valores. 
Los valores pueden ser de cualquier tipo, y están indexados por enteros, por lo que las tuplas son muy parecidas a listas. 
La diferencia importante es que las tuplas son inmutables.

ej:
```python
t1 = (1, 2, 3, 4, 5)
t2 = (6, 7, 8 ,9, "-_-")
t3 = t1 + t2

print("l3:", t3)

#Deberia tirar un error debido a que se esta modificando un valor de la tupla!
t1[0] = "TPS"

#Output: l3: (1, 2, 3, 4, 5, 6, 7, 8, 9, '-_-')
#Output: 
#---------------------------------------------------------------------------
#TypeError                                 Traceback (most recent call last)
#/tmp/ipython-input-746852891.py in <cell line: 0>()
#     10 
#     11 #Deberia tirar un error debido a que se esta modificando un valor de la tupla!
#---> 12 t1[0] = "TPS"                                                                   #podria asignar nueva memoria, seria otra tupla: t1 = ("TPS",) + t1[1:]
#
#TypeError: 'tuple' object does not support item assignment
```

Nota que las tuplas son inmutables, por lo que no se pueden modificar sus valores, de intentarlo lanzará un error.
""")

st.markdown("### Diccionarios")
st.markdown("""
Un diccionario es como una lista, pero más general. 
En una lista, los índices deben ser enteros; En un diccionario pueden ser (casi) cualquier tipo. 

**características**
    * Tiene estructura *clave*:*valor*
    * La *clave* debe ser única
    * No existe orden y se accede al valor mediante la clave
    * Para añadir elementos a un diccionario usaremos corchetes {}
    * Tiempo constante.

ej 1:
```python
diccionario = {"clave1": "valor1", "clave2": 2.0, "clave3": 3}
print(diccionario)
#Output: {"clave1": "valor1", "clave2": 2.0, "clave3": 3}
```
Si quiero obtener el valor dada una clave debo obtenerlo tal como lo haría con una lista.

ej 2:
```python
print(diccionario["clave1"])
#Output: valor1
```
Aunque al tener cierta flexibilidad con las claves no se puede acceder a un valor si no existe.
esto podría generar problemas en un código por lo que la forma segura de extraer un valor es usando el método `get()`.

ej 3:
```python
print(diccionario.get("clave1", "Valor por defecto"))
print(diccionario.get("clave", "valor por defecto"))

#Output: valor1
#Output: valor por defecto
```
y si no se le pone ningún valor suele retornar None.

ej 4:
```python
print(diccionario.get("clave4"))
#Output: None
```

Bien, ahora veamos algunos métodos útiles de los diccionarios.

**métodos**
* `keys()` - Retorna las claves del diccionario
* `values()` - Retorna los valores del diccionario
* `items()` - Retorna pares clave-valor del diccionario
* `get(key, default)` - Retorna el valor de la clave o un valor por defecto
* `pop(key)` - Elimina y retorna el valor de la clave
* `update(other_dict)` - Actualiza el diccionario con otro diccionario

ej 5:
```python
print(diccionario.keys())
#Output: dict_keys(['clave1', 'clave2', 'clave3'])
```

ej 6:
```python
print(diccionario.values())
#Output: dict_values(['valor1', 2.0, 3])
```

ej 7:
```python
print(diccionario.items())
#Output: dict_items([('clave1', 'valor1'), ('clave2', 2.0), ('clave3', 3)])
```

ej 8:
```python
print(diccionario.get("clave1"))
#Output: valor1
```

**Sets**Un set es una estructura de datos mutable (como las listas y los diccionarios), que permite
agregar y quitar elementos cumpliendo los requisitos de unicidad y búsqueda en tiempo constante. Además es posible hacer operaciones entre sets como unión, intersección y diferencia
muy fácilmente:

* No tienen orden
* No permiten duplicados
* Se accede a los elementos de forma indirecta

> Nota: se puede decir que es un caso particular de los diccionarios en donde no tenemos values, solo keys.

ej :

```python
s1 = {1, 2, 3, 4}
print(s1)
# Output: {1, 2, 3, 4}

s1.add(1)
print(s1)
# Output: {1, 2, 3, 4}

s2 = {3, 4, 5, 6}
print(s1.union(s2))
#Output: {1, 2, 3, 4, 5, 6}

print(s1.intersection(s2))
#Output: {3, 4}

print(s1.difference(s2))
#Output: {1, 2}

```
""")


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 