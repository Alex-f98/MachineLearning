import streamlit as st
from st_pages import add_page_title, hide_pages

add_page_title(layout="wide")

hide_pages(["Thank you"])

st.markdown("""
Indice:

* Numpy
* Matrices
* Operaciones con matrices
* Vectorixación
---
""", unsafe_allow_html=True)

st.markdown("### [Numpy](https://www.datacamp.com/tutorial/python-numpy-tutorial)")
st.markdown("""
**NumPy** (Numerical Python) es una biblioteca fundamental para la computación científica en Python. 
Proporciona soporte para arrays multidimensionales y matrices, junto con una colección de funciones matemáticas de alto nivel para operar con estos arrays.

Caracteristicas:

* Eficiente
* **Multidimensional**
* Operaciones con vectores
* Compatible con muchas bibliotecas


**¿Pero qué es un arreglo?**

Cuando miras la impresión de **un par de arrays**, puedes verla como una cuadrícula que contiene valores del mismo tipo:

```python
import numpy as np
# Define un 1D array
array_1d = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8]],
                    dtype=np.int64)
# Define un 2D array
array_2d = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8]],
                       dtype=np.int64)
# Define un 3D array
array_3d = np.array([[[1, 2, 3, 4],
                         [5, 6, 7, 8]],
                        [[1, 2, 3, 4],
                         [9, 10, 11, 12]]],
                       dtype=np.int64)

print("Array 1D:")
print(array_1d)

print("Array 2D:")
print(array_2d)

print("Array 3D:")
print(array_3d)

# Output:
# Array 1D:
# [[1 2 3 4]
#  [5 6 7 8]]

# Array 2D:
# [[1 2 3 4]
#  [5 6 7 8]]

# Array 3D:
# [[[ 1  2  3  4]
#   [ 5  6  7  8]]
#
#  [[ 1  2  3  4]
#   [ 9 10 11 12]]]
```

Bueno, osea que practicamente puedo armar un array de cualquier dimension con listas de listas.

Cada nivel de anidamiento representa una dimension adicional.


En el ejemplo anterior, ves que los datos son enteros. 
El arreglo (array) almacena y representa cualquier tipo de datos regulares de manera estructurada.  

Sin embargo, debes saber que, a nivel estructural,**un arreglo no es más que un conjunto de punteros**. 
Es una combinación de una dirección de memoria, un tipo de dato, una forma (*shape*) y unos pasos (*strides*):  

- **El puntero de datos** indica la dirección de memoria del primer byte en el arreglo.  
- **El tipo de dato (dtype)** describe la clase de elementos que contiene el arreglo.  
- **La forma (shape)** indica la estructura o dimensiones del arreglo.  
- **Los pasos (strides)** son la cantidad de bytes que deben saltarse en memoria para llegar al siguiente elemento.  
  - Por ejemplo, si tus *strides* son `(10, 1)`, necesitas avanzar un byte para llegar a la siguiente columna y 10 bytes para localizar la siguiente fila.  

En otras palabras, un arreglo contiene información sobre los datos en bruto, cómo localizar un elemento y cómo interpretarlo.  

""")

st.markdown("""
Puedes pensar en estos arreglos multidimensionales como una especie de cajón que contiene cajones más pequeños, y así sucesivamente.
""")
st.image("python_course/image/NumpyArrays.png", caption="Representación visual de un arreglo NumPy", width=800)

st.markdown("""
Bueno, algo de lo que se suele hacer con numpy.

Puedo definir una lista de valores para operar.

```python
  import numpy as np

  l5 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  #Output: [ 1  2  3  4  5  6  7  8  9 10]
  ```

  Tambien puedo crearlas definiendo un rango de valores equiespaciados

  ```python
  l6 = np.linspace(0, 10, 5)
  #Output: [ 0.   2.5  5.   7.5 10. ]
  ```

Puedo crear arreglos con valores aleatorios.

```python
  l7 = np.random.random((2, 3))
  #Output: [[0.12345678 0.23456789 0.34567891]
  #         [0.45678912 0.56789123 0.67891234]]
```

O puedo crear arreglos y usarlos para iterar.
```python
  for valor in np.linspace(0, 1, 3):
    print(f"valor: {valor}")

  #Output: 
  #valor: 0.0
  #valor: 0.5
  #valor: 1.0
```
Si quiero iterar y obtener el índice y el valor:
```python
for i, valor in enumerate(np.linspace(0, 1, 3)):
  print(f"i: {i}: valor: {valor}")
```
#Output: 
#i: 0: valor: 0.0
#i: 1: valor: 0.5
#i: 2: valor: 1.0


Puedo crear matrices, nultiplicarlas por un escalar, y operarmatricialmente!

```python
# Puedo concatenar
X = 2*np.ones((3,3))
b = np.ones((3,1))

print("X:", X)
print("b:", b)

#Output:
#X: [[2. 2. 2.]
#    [2. 2. 2.]
#    [2. 2. 2.]]
#b: [[1.]
#    [1.]
#    [1.]]


#concateno por columnas:
print(np.c_[X , b ])          #np.concatenate((X,b), axis = 1)
print("\n")

#Output:
#[[2. 2. 2. 1.]
# [2. 2. 2. 1.]
# [2. 2. 2. 1.]]

#concateno por filas
print(np.r_[X, b.T])          #np.concatenate((X,b.T), axis = 0)

#Output:
#[[2. 2. 2.]
# [2. 2. 2.]
# [2. 2. 2.]
# [1. 1. 1.]]
```

> Nota: `np.c_[]` concatena por columnas y `np.r_[]` concatena por filas es una forma abreviada de `np.concatenate()`.

Puedo realizar distintas operaciones matematicas  incluso mesclarlas con slicing.
```python
#Elevar al cuadrado los numeros pares.
#l5[l5%2==0]
print("l5: ", l5)
print("l5**2: ", l5**2)
print("l5%2==0: ", l5%2==0)
print("l5[l5%2==0]**2: ", l5[l5%2==0]**2 )`

#Output:
#l5:  [ 1  2  3  4  5  6  7  8  9 10]
#l5**2:  [  1   4   9  16  25  36  49  64  81 100]
#l5%2==0:  [False  True False  True False  True False  True False  True]
#l5[l5%2==0]**2:  [  4  16  36  64 100]
```

> Nota: Las ultimas dos lineas mustran como defino algo conocido como mascara de booleanos.
> Esa mascara se puede usar para filtrar los elementos de un array como se muestra en la ultima linea.

Puedes observar que las operaciones que se nos permite hacer son operaciones matematicas, y recordar que las operaciones se realizan elemento a elemento.

```python
#Suma de vectores
np.array([1,2,3]) + np.array([1,1,1])
#Output: [2 3 4]

#Producto elemento a elemento
np.array([1,2,3]) * np.array([2,2,2])
#Output: [2 4 6]

#Producto por un escalar.
# l5:  [ 1  2  3  4  5  6  7  8  9 10]
2 * l5
#Output: [ 2  4  6  8 10 12 14 16 18 20]
```
> Nota: Las sumas en listas eran equivalente a la concatenacion, aquí eso no tendria sentido.

**Producto escalar y matricial**

``` np.dot() ```
Producto escalar o producto punto:

$$
a.b = \sum_{i=1}^n a_i b_i
$$

Internamente esto usa la multiplicacion matricial `np.matmul()`
```python
#Producto punto(internamente es un matmul)
np.dot(np.array([1, 2, 3]), np.array([2, 2, 2])) # Esto hacerlo con @
#Output: 12
```

``` np.matmul() ```
Producto matricial

$$
C = A.B \\
c_{ij} = \sum_{k=1}^n A_{ik} B_{kj}
$$


❗ Es equivalente usar $C = a@b$

```python
#producto matricial
a = np.array([[1, 0],
              [0, 1]])

b = np.array([[4, 1],
              [2, 2]])

print(f"a.shape: {a.shape}")
print(f"b.shape: {b.shape}")
#Output: a.shape: (2, 2)
#        b.shape: (2, 2)

C = np.matmul(a,b)
#Esto es lo mismo que:
C_ = a @ b
print(f"\n np.matmul(a,b):\n{C}, shape: {C.shape}")
print(f"\n a @ b:\n {C_} , shape: {(C_).shape}")

#Output:
# np.matmul(a,b):
# [[4 1]
#  [2 2]], shape: (2, 2)
#
# a @ b:
#  [[4 1]
#  [2 2]] , shape: (2, 2)
```
> Nota, se cumplen las dimensiones del producto matricial y el resultado es una matriz de 2x2.

> Un metodo interezante.
> En la matriz 'b',donde un valor en ``` 'b[ij] == 1' -> 'b[ij] <- 1' sino 'b[ij]<- 0' ```
> ```python
> np.where(b == 1, 5, 0)
> ```
> Output: [[0 5]
>          [0 0]]
""")

st.markdown("Producto tensorial")
#https://i2tutorials.com/what-do-you-mean-by-tensor-and-explain-about-tensor-datatype-and-ranks/
st.markdown(
    """ 
    Los tensores se pueden pensar como la extension de matrices a mas dimensiones.
    Internamente, son listas de listas de listas... y asi sucesivamente pero es mejor pensarlo como lista de matrices.

    Nos va a interezar como realizar multiplicaciones de las matrices contenidas en estos tensores
    """
)
st.image("python_course/image/tensor.png", caption="Representación visual de un tensor", width=800)
st.markdown(
    """
    > Nota 1: Es muy importante saber que que en el producto tensorial la multiplicacion se va a dar a lo largo de las ultimas 2 dimensiones.
    
    > Nota 2: [Estrictamente no son tensores](https://towardsdatascience.com/what-is-a-tensor-in-deep-learning-6dedd95d6507/)
    """
    )
st.code("""
    # Operacion con tensores
    a = np.array([
                [[1, 2],
                [3, 4]]
                    ,
                [[5, 6],
                    [7, 8]]
                ])              #shape: (2, 2,2)
    #Observa que es una lista de matrices!


    b = np.array([
                [[1, 0],
                [0, 1]]
                    ,
                [[2, 0],
                [0, 2]]
                ])              #shape: (2, 2,2)

    c = np.array([[2, 0],[0, 2]]) #shape (2,2)

    #Notar como estan puestas las dimensiones!!!

    print(f"a.shape: {a.shape}")
    print(f"b.shape: {b.shape}")
    print(f"c.shape: {c.shape}")

    #Output:
    # a.shape: (2, 2, 2)
    # b.shape: (2, 2, 2)
    # c.shape: (2, 2)

    #Siendo N = 2 podemos pensar que dimension deberiamos obtener.
    #(N, 2,2)@(N, 2,2) --> (N, 2,2)
    print(f"\ n a @ b:\ n {a @ b} , shape: {(a @ b).shape}")
    #Output:
    # a@b:
    # [[[ 7 10]
    #   [15 22]]
    #
    #  [[31 38]
    #   [47 58]]] , shape: (2, 2, 2)

    #(N, 2,2)@(2,2)    --> (N, 2,2)
    print(f"\ n a @ c:\ n {a @ c} , shape: {(a @ c).shape}")
    #Output:
    # a@c:
    # [[[ 2  4]
    #   [ 6  8]]
    #
    #  [[10 12]
    #   [14 16]]] , shape: (2, 2, 2)
    """, language="python")

st.markdown("""
    > Nota: Observe como la primer dimensión es el largo de la lista de matrices y las demas dimensiones son el resultado de la multiplicacion de las matrices.

    Pero esto no trae algún problema si queremos realizar operacion con la traspuesta?
    """)
st.code("""
    d = np.array([[[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]],

                [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]])   #(2 ,3,3)

    e = d.T                      #(3, 3,2)
    """, language="python")

st.markdown("""
    vemos que las dimensiones traspuestas no son a lo largo de las 2 últimas dimensiones, sino que es para el tensor completo.

    > Recuerda que las multiplicaciones son miembro a miembro, eso quiere decir que cantidad (el largo) de elementos (matrices) en el tensor deben ser las mismas.
  """)

st.code("""
    #Notar que tiene un corchete de mas, eso agrega una dimensión.
    f = 2*np.array([[
                [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]
                ]])             #(1, 3,3)

    print(f"d.shape:   {d.shape}")
    print(f"d.T.shape: {e.shape}")
    print(f"f.shape:   {f.shape}")

    #Output:
    # d.shape:   (2, 3, 3)
    # d.T.shape: (3, 3, 2)
    # f.shape:   (1, 3, 3)

    #(2, 3, 3) @ (1, 3, 3)
    print(f"\n d @ f:\n {d @ f} , shape: {(d @ f).shape}")
    #Output:
    #d @ f:
    # [[[[2 0 0]
    #   [0 2 0]
    #   [0 0 2]]
    #
    #  [[2 0 0]
    #   [0 2 0]
    #   [0 0 2]]]] , shape: (1, 2, 3, 3)
    """, language="python")

st.markdown("""
    Osea, nos realiza el prodcuto pero me cambia las dimensiones, y lo peor es que **no falla** esto es un resultado inesperado.
    si no se prestya atención a las operaciones que se estan realizando, puede pasar desapercibido y generar problemas sin explicacion mas adelante.

    Volvamos a la traspuesta, aquí la forma correcta de hacer una trasposición con tensores es usar **np.transpose(array, axes)**
    el cual le decimos a que ejes queremos transponer y como.
  """)


st.code(""" 
    # Definimos N matrices 2x2
    N = 3
    X     = np.random.rand(N, 2, 2)          # (N, 2,2)
    Sigma = np.random.rand(N, 2, 2)          # (N, 2,2)

    # Transponemos solo las últimas dos dimensiones
    X_T = np.transpose(X, (0, 2, 1))         #eje (0,1,2) -> (0, 2,1)

    print("X:")
    print(X)
    print("\nX_T:")
    print(X_T)

    #Output:
    #X:
    # [[[0.5488135  0.71518937]
    #  [0.60276338 0.54488318]]
    #   
    # [[0.4236548  0.64589411]
    #  [0.43758721 0.891773  ]]
    #          ...
    # [[0.96366276 0.38344152]
    #  [0.79172504 0.52889492]]]

    #X_T:
    # [[[0.5488135  0.60276338]
    #  [0.71518937 0.54488318]]
    #
    # [[0.4236548  0.43758721]
    #  [0.64589411 0.891773  ]]
    #          ...
    # [[0.96366276 0.79172504]
    #  [0.38344152 0.52889492]]]
    """, language="python"
    )

st.markdown("[Vectorización](https://migue8gl.github.io/2026/01/06/vectorizacion-en-python.html) - El poder del [@](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)")
st.markdown(
    """
    *Utilizamos esta sección para explicar que es necesario vectorizar cuando es posible para mejorar el rendimiento de las operaciones.*

    "Es un método por el cual aplicamos operaciones sobre un conjunto de elementos de forma simultánea,
     en vez de aplicarlas de una en una. 
     Sencillo de entender y muy fácil de aplicar en Python con la librería de cálculo numérico NumPy"   
    """
    )
st.image("python_course/image/vectorization.png", caption="Representación visual de vectorización", width=800)

st.markdown(
    """
    Ok, ahora que entendemos que es la vectorización, veamos como aplicarla.
    Cuando que quieren hacer operaciones con las matrices internas lo usual es recorrer todas las matrices internas y operar con ellas una a una.

    siendo

    $P = [M_1,\  M_2, ... ,\  M_n]$ quiero realizar la siguiente operación a cada miembro de P.

    $P\ x\ M = [M_1, M_2, ... , M_n]\ x\ M$

    Lo que implica recorrer cada uno de los elementos y realizar la operación.

    $ P[1]\ x\ M = M_1\ x\ M$

    $ P[2]\ x\ M = M_2\ x\ M$

    ...

    $ P[n]\ x\ M = M_n\ x\ M$

    Para operaciones tensoriales con matrices muy grandes(imágenes) recorrer cada elemento implicaría traer a memoria la matriz $M_i$ y realizar la operación correspondiente para luego traer la matriz $M_{i+1}$ y realizar nuevamente una operación.

    El traer una a una las matrices a memoria conlleva tiempo, que sumado a un set muy grande de matrices el tiempo se vuelve apreciable por lo que la solución es **traer todo a memoria (siempre que la RAM te lo permita)** y operar directamente.

    Esto se logra con la operación tensorial ( @ ).

    Veamos como funciona esto, observemos los tiempos:
    ```python
    # Crear dos conjunto grande de matrices (100x100)
    N = 20_000
    P = rand(N, 100, 100)
    M = rand(N, 100, 100)


    # Multiplicación con el operador @
    start_tensorial  = time()
    result_tensorial = P @ M
    end_tensorial    = time()

    print(f"Tiempo de ejecución con @: {end_tensorial - start_tensorial:.4f} segundos")

    # Multiplicación manual con bucles
    start_manual  = time()
    result_manual = np.array([np.dot(P[i], M[i]) for i in range(N)])
    end_manual    = time()

    print(f"Tiempo de ejecución manual: {end_manual - start_manual:.4f} segundos")


    # Verificación de que ambos métodos producen el mismo resultado
    print("Los resultados son iguales:", np.allclose(result_manual, result_tensorial))

    # Output: 
    # Tiempo de ejecución con @:  3.5339 segundos
    # Tiempo de ejecución manual: 11.1785 segundos
    # Los resultados son iguales: True
    ```
    > Nota: vemos que los resultados son iguales pero el tiempo de ejecución con @ es mucho menor.
    > Quieres ver una implementación mas realista: [Aqui se aplica a algotirmos geneticos](https://migue8gl.github.io/2026/01/06/vectorizacion-en-python.html)

    """

)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 