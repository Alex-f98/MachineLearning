import streamlit as st
from st_pages import add_page_title, hide_pages

add_page_title(layout="wide")

hide_pages(["Thank you"])

st.markdown("""
Indice:

* Listas por comprensión.
* Segmentación de cadenas.
---
""", unsafe_allow_html=True)

st.markdown("### Listas por comprensión")
st.markdown("""

Se utiliza cuando se quiere aplicar una función a cada elemento de la lista cambiando la sintaxis.

Se mete el for dentro de la lista

```python
[<expresión> for <variable> in <secuencia>]
```

Si hay una condición se puede agregar.
```python
[<variable> for <variable> in <secuencia> if <condicion>]
```

La sintaxis se puede extender a diccionarios o tuplas.
>Nota: La sintaxis usando solo un bucle for sería:
>```python
>for <variable> in <secuencia>:
>    <expresión>
>```
> 

ej1 : listas
```python
numeros = [1, 2, 3, 4, 5]
cuadrados = [x**2 for x in numeros]
print(cuadrados)
#Output: [1, 4, 9, 16, 25]
```

Esto se puede generalizar a más estructuras como por ejemplo un diccionario

ej 2: diccionarios
```python
numeros = [1, 2, 3, 4, 5]
cuadrados = {x: x**2 for x in numeros}
print(cuadrados)
#Output: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
```
""")

st.markdown("### Segmentación de cadenas (List [Slicing](https://www.datacamp.com/tutorial/python-slice))")
st.markdown("""

La sintaxis de corte es la forma más común de acceder a partes de una secuencia. Cada parámetro controla cómo se realiza el segmentado: **sequence[start:stop:step]** donde *start*, *stop* y *step* son los parámetros.

* **start**: Índice donde comienza la porción (inclusive). Por defecto se omite en 0.
* **stop**: Índice donde termina la porción (exclusivo). Por defecto se mantiene en la duración de la secuencia si se omite.
* **step**: Determina el intervalo entre los elementos. Por defecto se pone en 1 si se omite


```python
numbers = [10, 20, 30, 40, 50, 60]  

print(numbers[1:4])  
# Output: [20, 30, 40]  

print(numbers[:3])   
# Output: [10, 20, 30]  

print(numbers[::2])  
# Output: [10, 30, 50]  
```
> Nota: esto último se lee como "desde el inicio hasta el final en pasos de 2"

También puedes invertir una lista iterando con paso de -1.
```python
numbers = [10, 20, 30, 40, 50, 60]  
print(numbers[::-1]) 
# Output: [60, 50, 40, 30, 20, 10]  
```
> Nota: esto se lee como "desde el inicio hasta el final en pasos de -1"



**Ventajas principales del slicing en Python**

- **Código más limpio y conciso**  
  Permite extraer, reorganizar o filtrar datos sin necesidad de bucles ni condiciones complejas. 

- **Procesamiento eficiente de datos**  
  Está optimizado internamente para ser rápido y ahorrar memoria. Ideal para trabajar con grandes volúmenes de información (ej. logs, datasets).

- **Versatilidad en múltiples estructuras**  
  Funciona en listas, tuplas, cadenas, rangos y se extiende a librerías como **NumPy** y **pandas**, donde se vuelve aún más poderoso.

- **Aplicaciones prácticas en el mundo real**  
  - Extraer columnas específicas de un DataFrame.  
  - Tomar las últimas N entradas de un archivo de registros.  
  - Invertir cadenas o listas rápidamente.  
  - Seleccionar subconjuntos de datos para análisis.

- **Eficiencia en librerías científicas**  
  - En **NumPy**, el slicing devuelve *views* (referencias al mismo bloque de memoria), lo que evita copias innecesarias y acelera cálculos.  
  - En **pandas**, se integra con `.loc[]` y `.iloc[]` para un control preciso sobre filas y columnas.

- **Manipulación avanzada**  
  Con slicing puedes reemplazar, insertar o eliminar múltiples elementos de una lista en una sola línea, lo que simplifica operaciones de edición.

> Nota: Cuando se hace slicing se crea una nueva lista con referencia a los elementos originales, no una copia completa.
>
> ```python
> lista = [[1], [2], [3], [4]]
> sub = lista[1:3]
> sub[0].append(99)
> print(lista)  
> #Output: [[1], [2, 99], [3], [4]]
> ```

**Slicing con numpy**
Los arreglos NumPy llevan el slice al siguiente nivel, 
ofreciendo herramientas potentes para manipular grandes conjuntos de datos multidimensionales. 
Una diferencia crítica en el segmentado NumPy son vistas frente a copias: 
cortar un array NumPy normalmente devuelve una vista (una referencia a los datos originales), 
no una copia nueva. Este comportamiento garantiza eficiencia al trabajar con grandes 
conjuntos de datos, pero requiere un manejo cuidadoso para evitar cambios no intencionados.


```python
import numpy as np  

# Create a 1D array
array = np.array([10, 20, 30, 40, 50])

# Slice elements from index 1 to 3
print(array[1:4])  
# Output: [20 30 40]  

# Apply step
print(array[::2])  
# Output: [10 30 50]  

```
Pero qué es Numpy y qué son los arrays? lo veremos en el siguiente módulo.

""")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 