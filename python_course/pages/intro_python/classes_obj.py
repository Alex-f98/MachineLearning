import streamlit as st

st.markdown("""
Índice:

* [Funciones](#funciones)
* [Objetos](#objetos)
* [Polimorfismo y herencia](#polimorfismo-y-herencia)
---
""", unsafe_allow_html=True)

st.markdown("### Funciones")
st.markdown(
    """
    Una función en Python es un bloque de código identificado por un nombre, que encapsula una tarea específica y puede ser reutilizado.
    Se define con la palabra clave def, seguida del nombre de la función, una lista de parámetros entre paréntesis y un cuerpo de instrucciones.
    Opcionalmente, puede devolver un resultado mediante la instrucción return.
     
    ```python
    def nombre_funcion(param1, param2, ...):
        '''
        Documentación de la función (docstring).
        Explica qué hace, qué parámetros recibe y qué devuelve.
        '''
        # cuerpo de la función
        return resultado
    ```
    Ventajas de usar una función:
    - Reutilización de código: se evita repetir instrucciones en distintos lugares del programa.
    - Modularidad: el programa se divide en partes más pequeñas y comprensibles.
    - Legibilidad: el código se vuelve más claro y fácil de mantener.
    - Abstracción: permite ocultar detalles de implementación y enfocarse en la tarea que realiza.
    - Facilidad de prueba: cada función puede probarse de manera independiente.
    - Colaboración: en proyectos grandes, distintos programadores pueden trabajar en funciones específicas sin interferir entre sí.

    ej:
    ```python
    def area_rectangulo(base, altura):
        '''Devuelve el área de un rectángulo.'''
        return base * altura

    def area_circulo(radio):
        '''Devuelve el área de un círculo.'''
        import math
        return math.pi * radio**2

    def mostrar_area_circulo(radio):
        '''Imprime directamente el área de un círculo.'''
        print("El área del círculo es:", math.pi * radio**2)

    ```

    """)


st.markdown("### Objetos")
st.markdown(
    """
    *Los objetos son una manera de organizar datos y de relacionar esos datos con el código apropiado para manejarlo. 
    Son los protagonistas de un paradigma de programación llamado Programación Orientada a Objetos.*
    
    *Nosotros ya usamos objetos en Python sin mencionarlo explícitamente. Es más, todos los
    tipos de datos que Python nos provee son, en realidad, objetos*

    **Pero bien, más formalmente ¿qué son los objetos?**

    En Python, todos los tipos son objetos. Pero no en todos los lenguajes de programación es
    así. En general, podemos decir que un objeto es una forma ordenada de agrupar datos (los
    atributos) y operaciones a utilizar sobre esos datos (los métodos).

    Es importante notar que cuando decimos objetos podemos estar haciendo referencia a dos
    cosas parecidas, pero distintas.
    Por un lado, la definición del tipo, donde se indican cuáles son los atributos y métodos que
    van a tener todas las variables que sean de ese tipo. Esta definición se llama específicamente, la
    clase del objeto.
    
    **A partir de una clase es posible crear distintos valores que son de ese tipo**. A cada uno de
    los valores generados a partir de una clase se los llama instancia de esa clase.

    Se dice que los objetos tienen estado y comportamiento, ya que los valores que tengan los atributos
    de una instancia determinan el estado actual de esa instancia, y los métodos definidos en una clase
    determinan cómo se va a comportar ese objeto.

    Ok, pero veamos unos ejemplos prácticos para cerrar conceptos:

    Ejemplo 1: ya usábamos objetos y no sabíamos.
    ```python
    lista = [1,2,3,4,"f"]
    print(type(lista))
    #Output: <class 'list'>  #en nuevas versiones de Python solo aparece "list".
    ```
    Es justamente una instancia de la clase "list", el objeto es la variable lista.

    Ejemplo 2: Podemos ver qué métodos y atributos tienen.
    ```python
    lista = [1,2,3,4,"f"]
    print(dir(lista))
    #Output: ['__add__', 
    #         '__class__', 
    #         '__class_getitem__', 
    #         ..., 
    #         'append', 
    #         'clear', 
    #         'copy', 
    #         'count', 
    #         'extend', 
    #         'index', 
    #         'insert', 
    #         'pop', 
    #         'remove', 
    #         'reverse', 
    #         'sort']
    ```
    Nos da el listado de atributos y métodos todos mezclados, los últimos son métodos de la clase "list".
    Aquellos atributos que empiezan y terminan con "__" son atributos especiales que Python usa internamente para definir el comportamiento interno.

    Ejemplo 3: Un atributo es un valor que tiene un objeto.
    ```python
    lista = [1,2,3,4,"f"]
    print(lista.__len__())
    #Output: 5
    ```
    > Nota: ```python__len__``` es un método que devuelve la cantidad de elementos que tiene la lista y de hecho es el método que se llama cuando usamos la función len()
    
    Ejemplo 4: Un método es una operación que puede realizar un objeto.
    ```python
    lista = [1,2,3,4,"f"]
    lista.append(5)           #agrega el elemento 5 a la lista
    print(lista)
    #Output: [1, 2, 3, 4, 'f', 5]
    ```
    Bien, ahora que vimos que los objetos tienen estado y comportamiento, veamos cómo se definen.

    ```python
    class MiClase:
        def __init__(self, atributo):
            self._atributo = atributo
    ```
    La sintaxis "self._atributo = atributo" es la forma de definir un atributo en una clase.
    Simplemente, "self" es una convención que indica que el atributo pertenece a la instancia de la clase y puedo asignarle un valor que luego podré usar en los métodos de la clase.
    
    Aquí la función def __init__ es el constructor de la clase, es decir, es la función que se llama cuando se crea una instancia de la clase.

    Sigamos completándolo un poco más.
    ```python
    class MiClase:
        def __init__(self, atributo):
            self._atributo = atributo

        def get_atributo(self):
            return self._atributo
    ```

    Vemos que ahora existen dos funciones dentro de mi clase def __init__ que es el constructor de la clase y *get_atributo* que es un método de la clase.
    este método debe ser llamado para obtener el valor del atributo (no se llama automáticamente).
    Ejemplo 5:
    ```python
    mi_clase = MiClase("Soy un atributo_777 :v")
    print(mi_clase.get_atributo())
    #Output: Soy un atributo_777 :v
    ```

    Bien, pero el método "append()" recibía un parámetro, entonces también se puede usar en la definición de un método.
    ```python
    def nombre_metodo(self, parametro1, parametro2, ...):
        # codigo del metodo
    ```
    Ejemplo 6:
    ```python
    class MiClase:
        def __init__(self, atributo):
            self._atributo = atributo

        def get_atributo(self, parametro1):
            return self._atributo + parametro1

    mi_clase = MiClase("Soy un atributo")
    print(mi_clase.get_atributo(" y yo un parametro :v"))
    #Output: Soy un atributo y yo un parámetro :v
    ```

    Listo, ahora que vimos cómo se definen las clases y los objetos hagamos algo un poquito más serio.

    Ejemplo 7: Creemos un Therian.
    ```python
    class Therian():
        def __init__(self,
                    nombre  : str,           #estoy indicando que es tipo string.
                    especie : str,    
                    edad    : int):          #estoy indicando que es tipo entero.
            self.nombre         = nombre
            self.especie        = especie
            self.edad           = edad
            self.estado         = "humano"
            self.habilidades    = []          #creo una lista vacía para guardar las habilidades del therian.
            self.nivel_energia  = 100     
        
        def transformar(self):
            if self.estado == "humano":
                self.estado = "animal"
            else:
                self.estado = "humano"
            print(f"- {self.nombre} se ha transformado en {self.estado}.")

        def agregar_habilidad(self, habilidad):
            self.habilidades.append(habilidad)
            print(f"- Habilidad '{habilidad}' añadida a {self.nombre}.")

        def usar_habilidad(self, habilidad):
            '''Usar una habilidad cuesta 10 de energia'''
            if habilidad in self.habilidades and self.nivel_energia >= 10:
                self.nivel_energia -= 10
                print(f"- {self.nombre} usa la habilidad '{habilidad}'. Energía restante: {self.nivel_energia}")
            else:
                print(f"- {self.nombre} no puede usar '{habilidad}'.")

        def descansar(self):
            ''' Descansar recupera el 100% de la energía'''
            self.nivel_energia = 100
            print(f"- {self.nombre} ha descansado y recuperado toda su energía.")

        def presentarse(self):
            print(f'''- Soy {self.nombre}, un therian de especie {self.especie}, edad {self.edad}, estado actual: {self.estado}.
            Suelo habitar el barrio chino y si me ves, ¡corre!''')


    luna = Therian("Luna", "lobo", 21)
    luna.presentarse()
    luna.agregar_habilidad("Aullar a la luna")
    luna.transformar()
    luna.usar_habilidad("Aullar a la luna")
    luna.descansar()

    #Output:
    #- Soy Luna, un therian de especie lobo, edad 21, estado actual: humano.
    #- Suelo habitar el barrio chino y, si me ves, ¡corre!
    #- Habilidad 'Aullar a la luna' añadida a Luna.
    #- Luna se ha transformado en animal.
    #- Luna usa la habilidad 'Aullar a la luna'. Energía restante: 90
    #- Luna ha descansado y recuperado toda su energía.
    ```

    > Nota: Hace falta que lo explique? miralo con calma y entiende cada parte, es mas facil de lo que parece.


    **[Métodos especiales](https://allendowney.github.io/ThinkPython/chap16.html)**

    Son funciones definidas dentro de una clase que comienzan y terminan con `__`.  
    Python los usa para dar comportamiento especial a los objetos, como inicialización, representación en texto, comparación, operaciones matemáticas, etc.  


    ## Principales métodos especiales

    | Método | Uso principal | Ejemplo |
    |--------|---------------|---------|
    | `__init__(self, ...)` | Inicializa un objeto al crearlo. | `p = Point(0, 0)` |
    | `__str__(self)` | Devuelve una representación en texto “amigable” para `print()`. | `print(p)` → `Point(0, 0)` |
    | `__repr__(self)` | Representación más técnica, útil para depuración. | `repr(p)` |
    | `__eq__(self, other)` | Define cuándo dos objetos se consideran iguales (`==`). | `p1 == p2` |
    | `__lt__`, `__gt__`, etc. | Comparaciones de orden (`<`, `>`). | `p1 < p2` |
    | `__add__`, `__sub__`, etc. | Operadores aritméticos personalizados. | `p1 + p2` |
    | `__len__(self)` | Define el comportamiento de `len(obj)`. | `len(lista)` |
    | `__getitem__`, `__setitem__` | Permiten que el objeto se comporte como una colección. | `obj[0]` |
    | `__call__(self, ...)` | Hace que el objeto sea “llamable” como una función. | `obj()` |

    ---

    ## Ventajas de usar métodos especiales
    - **Integración con el lenguaje**: tus clases se comportan como tipos nativos de Python.  
    - **Legibilidad**: objetos muestran información clara al imprimirlos.  
    - **Flexibilidad**: puedes redefinir operadores y comparaciones según tu modelo.  
    - **Polimorfismo**: diferentes clases pueden compartir métodos con el mismo nombre (`draw`, `__eq__`), lo que permite tratarlas de forma uniforme.  

    """)

st.markdown("### Polimorfismo y herencia")
st.markdown(
    """
    **Herencia**

    La herencia es un mecanismo de la programación orientada a objetos que permite crear nuevas clases basadas en clases existentes.
    La clase base (o superclase) define atributos y métodos comunes.
    La clase derivada (o subclase) hereda esos atributos y métodos, y puede extenderlos o sobrescribirlos.

    *Ventaja: reutilización de código y organización jerárquica.*

    **Polimorfismo**

    El polimorfismo significa “muchas formas”. En programación, se refiere a la capacidad de distintos objetos de responder de manera diferente a la misma operación o método.
    Ejemplo: varias clases pueden implementar un método presentarse(), pero cada una lo hace de manera diferente.
    Permite escribir código más general y flexible, que funciona con diferentes tipos de objetos sin necesidad de conocer sus detalles internos.
    
    ej:
    ```python
    class Lobo(Therian):
        def presentarse(self):
            '''Presentación específica para lobos.'''
            print(f"Auuuu! Soy {self.nombre}, un lobo therian de {self.edad} años. Actualmente estoy en forma {self.estado}.")

    class Felino(Therian):
        def presentarse(self):
            '''Presentación específica para felinos.'''
            print(f"Miau! Soy {self.nombre}, un felino therian de {self.edad} años. Mi estado actual es {self.estado}.")


    luna = Lobo("Luna", "lobo"  , 21)
    leo = Felino("Leo", "felino", 19)

    luna.presentarse()
    luna.descansar()

    leo.presentarse()
    leo.descansar()

    # Output:
    # - Auuuu! Soy Luna, un lobo therian de 21 años. Actualmente estoy en forma humano.
    #   Luna ha descansado y recuperado toda su energía.
    # - Miau! Soy Leo, un felino therian de 19 años. Mi estado actual es humano.
    #   Leo ha descansado y recuperado toda su energía.
    ```
    
    Bien, o sea en este ejemplo usamos la clase therian que habíamos definido antes como clase base o superclase.
    La notación es la siguiente:
    ```python
    class NuevaClase(ClaseBase):
        ...

    ```
    En este caso la clase "Lobo" hereda de la clase "Therian".
    ```python
    class Lobo(Therian):
        ...
    class Felino(Therian):
        ...
    ```
    esto significa que no se necesita repetir código que ya está en la clase base.

    Es decir métodos como *descansar* o *usar_habilidad* ya están definidos en la clase base "Therian".

    Pero vemos que solo sobreescribimos el método *presentarse* esto es porque cada subclase puede tener su propia forma de presentarse, así usamos el polimorfismo.
    
    
    
    """
)

st.markdown("### Ejemplo: Normalizacion (Scaler)")
st.markdown(
    """

    Bien, ahora veamos un ejemplo más real de herencia y polimorfismo.


    En Machine Learning, muchas técnicas requieren que las variables numéricas estén en una escala comparable. 

    Para esto, se utilizan scalers, que transforman los datos mediante estadísticas calculadas sobre el conjunto de entrenamiento (media, desviación estándar, mínimo, máximo, etc.).

    Implementar en Python una clase que realice escalado de datos numéricos, 
    inspirada en los scalers de sklearn.preprocessing, sin utilizar directamente dichas clases.

    Las clases deben:

    - Heredad de una clase Scaler base
    - Las clases hijas deben implementar los métodos *fit* y *transform*
    - Ajustarse a un conjunto de datos calculando los parámetros necesarios (fit)
    - Transformar datos usando esos parámetros (transform)
    - Ofrecer un método combinado (fit_transform)
    - El comportamiento debe ser equivalente, a nivel funcional, a StandardScaler y MinMaxScaler de sklearn.

    > Mas detalle lo puedes encontrar en la [documentación de sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).



    ```python
    class Scaler:
        def fit(self, X): #se puede poner un tipo de dato específico (self, X: pd.DataFrame|np.ndarray)
            '''
            Ajusta el scaler a los datos X.
            
            Args:
                X: DataFrame o array-like con los datos a escalar
            '''
            pass
        
        def transform(self, X):
            '''
            Transforma los datos X usando los parámetros ajustados.
            
            Args:
                X: DataFrame o array-like con los datos a transformar
                
            Returns:
                DataFrame o array-like con los datos transformados
            '''
            pass
        
        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)

        def inverse_transform(self, X):
            pass



    class MinMaxScaler(Scaler):
        def fit(self, X):
            self.min = X.min()
            self.max = X.max()
        
        def transform(self, X):
            return (X - self.min) / (self.max - self.min) #se podria validar la division por cero
        
        def inverse_transform(self, X):
            return X * (self.max - self.min) + self.min

    class StandardScaler(Scaler):
        def fit(self, X):
            self.mean = X.mean()
            self.std  = X.std()
        
        def transform(self, X):
            return (X - self.mean) / self.std

        def inverse_transform(self, X):
            return X * self.std + self.mean
    ```


    Bien, esto hay que explicarlo por que puede marear.

    primero, la funcion matematica de standard scaler es:

    $$
    z = \\frac{x - \mu}{\sigma}
    $$

    donde $\mu$ es la media y $\sigma$ es la desviacion estandar.

    y la funcion matematica de minmax scaler es:

    $$
    x' = \\frac{x - \min(x)}{\max(x) - \min(x)}
    $$

    donde $\min(x)$ es el valor minimo y $\max(x)$ es el valor maximo.

    estas funciones estan implementadas en sus respectivas funciones transform, que veremos mas en detalle.

    primero tenemos la clase padre **Scaler**, que es la que define los metodos que deben implementar 
    las clases hijas.
    
    Estas son:
    - `fit(self, X)`: Ajusta el scaler a los datos X.
    - `transform(self, X)`: Transforma los datos X usando los parámetros ajustados.
    - `fit_transform(self, X)`: Ajusta y transforma los datos X.
    - `inverse_transform(self, X)`: Invierte la transformación.

    Puedes notar que no es necesario una implementación concreta para los metodos de la clase padre,
    ya que las clases hijas van a implementarlos de manera diferente.

    Aunque, en el metodo **fit_transform**, podemos ver que no es necesario implementarlo en las clases hijas,
    ya que la clase padre lo implementa de manera general y puede ser llamado directamente.

    En **StandardScaler**, 
    - El metodo **fit** define dos nuevos atributos: **mean** y **std**, que son la media y la desviacion estandar de los datos
    las cuales no estaban definido en la clase padre, pero pueden ser accedidos desde los metodos **transform** y **inverse_transform** luego de que **fit** sea llamado.

    - El metodo **transform** aplica la funcion matematica de standard scaler a los datos X, estos pueden ser otros datos diferentes a los que se usaron en **fit**.
    - El metodo **inverse_transform** aplica la funcion matematica de inverse standard scaler a los datos X, estos pueden ser otros datos diferentes a los que se usaron en **fit**.

    En **MinMaxScaler**, 
    - El metodo **fit** define dos nuevos atributos: **min** y **max**, que son el valor minimo y el valor maximo de los datos
    las cuales no estaban definido en la clase padre, pero pueden ser accedidos desde los metodos **transform** y **inverse_transform** luego de que **fit** sea llamado.

    - El metodo **transform** aplica la funcion matematica de minmax scaler a los datos X, estos pueden ser otros datos diferentes a los que se usaron en **fit**.
    - El metodo **inverse_transform** aplica la funcion matematica de inverse minmax scaler a los datos X, estos pueden ser otros datos diferentes a los que se usaron en **fit**.

    > Se le pueden agregar distintas validaciones e incluso heredar de mas padres, para el ejemplo esto es suficiente.


    Puedes probarlo de la siguiente manera:

    ```python
    scaler_1 = StandardScaler()
    scaler_2 = MinMaxScaler()

    X_train_scaled_1 = scaler_1.fit_transform(X)
    X_train_scaled_2 = scaler_2.fit_transform(X)
    ```

    Bien, aqui scaler_1 y scaler_2 son dos objetos diferentes, cada uno con su propia instancia de la clase StandardScaler y MinMaxScaler.
    Cada uno de ellos tiene sus propios atributos y metodos, y pueden ser usados de manera independiente.

    llamo directamente a sus respectivos **fit_transform** para aplicar entrenar los parametros y luego aplica la transformación.
    este metodo no fue implementado en las clases hijas pero hereda el metodo de la clase padre que usa las implementaciones de **fit** y **transform**.
    
    > Es como si estuvieramos sobrescrbiendo los metodos fit y transform en las clases hijas.

    Puedes probar el codigo y ver su funcionamiento al final del siguiente [notebook](https://github.com/Alex-f98/MachineLearning/blob/main/notebooks/normalization_experiments.ipynb)

    

"""
)



hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 