import streamlit as st

st.markdown("""
Indice:

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

    **Pero bien, mas formalmente que son los objetos?**

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

    Ok, pero veamos un ejemplos practicos para cerrar conceptos:

    ej 1: ya usabamos objetos y no sabiamos.
    ```python
    lista = [1,2,3,4,"f"]
    print(type(lista))
    #Output: <class 'list'>  #en nuevas versiones de python solo aparece "list".
    ```
    Es justamente una instancia de la clase "list", el objeto es la variable lista.

    ej 2: Podemos ver que metodos y atributos tienen.
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
    Nos dá el listado de atributo y metodos todos mesclados, los ultimos son metodos de la clase "list".
    Aquellos atributos que empiezan y terminan con "__" son atributos especiables que python usa internamente para definir el comportamiento interno.

    ej 3: Un atributo es un valor que tiene un objeto.
    ```python
    lista = [1,2,3,4,"f"]
    print(lista.__len__())
    #Output: 5
    ```
    > Nota: __len__ es un metodo que devulve la cantidad de elementos que tiene la lista y de hecho es el metodo que se llama cuando usamos la funcion len()
    
    ej 4: Un metodo es una operacion que puede realizar un objeto.
    ```python
    lista = [1,2,3,4,"f"]
    lista.append(5)           #agrega el elemento 5 a la lista
    print(lista)
    #Output: [1, 2, 3, 4, 'f', 5]
    ```
    Bien, ahora que vimos que los objetos tienen estado y comportamiento, veamos como se definen.

    ```python
    class MiClase:
        def __init__(self, atributo):
            self._atributo = atributo
    ```
    La sintaxis "self._atributo = atributo" es la forma de definir un atributo en una clase.
    Simplemente, "self" es una convencion que indica que el atributo pertenece a la instancia de la clase y puedo asignarle un valor que luego podre usar en los metodos de la clase.
    
    Aquí la funcion def __init__ es el constructor de la clase, es decir, es la funcion que se llama cuando se crea una instancia de la clase.

    Sigamos completandolo un poco mas.
    ```python
    class MiClase:
        def __init__(self, atributo):
            self._atributo = atributo

        def get_atributo(self):
            return self._atributo
    ```

    Vemos que ahora existen dos funciones dentro de mi clase def __init__ que es el constructor de la clase y *get_atributo* que es un metodo de la clase.
    este metodo debe ser llamado para obtener el valor del atributo (no se llama automaticamente).
    ej 5:
    ```python
    mi_clase = MiClase("Soy un atributo_777 :v")
    print(mi_clase.get_atributo())
    #Output: Soy un atributo_777 :v
    ```

    Bien, pero el metodo "append()" recibia un parametro, entonces tambien se puede usar en la definicion de un metodo.
    ```python
    def nombre_metodo(self, parametro1, parametro2, ...):
        # codigo del metodo
    ```
    ej 6:
    ```python
    class MiClase:
        def __init__(self, atributo):
            self._atributo = atributo

        def get_atributo(self, parametro1):
            return self._atributo + parametro1

    mi_clase = MiClase("Soy un atributo")
    print(mi_clase.get_atributo(" y yo un parametro :v"))
    #Output: Soy un atributo y yo un parametro :v
    ```

    Listo, ahora que vimos como se definen las clases y los objetos hagamos algo un poquito mas serio.

    ej 7: Creemos un Therian.
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
            self.habilidades    = []          #creo una lista vacia para guardar las habilidades del therian.
            self.nivel_energia  = 100     
        
        def transformar(self):
            if self.estado == "humano":
                self.estado = "animal"
            else:
                estado = "humano"
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
            ''' Descansar recuepra el 100% de la energia'''
            self.nivel_energia = 100
            print(f"- {self.nombre} ha descansado y recuperado toda su energía.")

        def presentarse(self):
            print(f'''- Soy {self.nombre}, un therian de especie {self.especie}, edad {self.edad}, estado actual: {self.estado}.
            Suelo habitar el barrio chino y si me ves corre!.''')


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


    **[Metodos especiales](https://allendowney.github.io/ThinkPython/chap16.html)**

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

st.markdown("Polimorfismo y herencia")
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
    
    Bien, osea en este ejemplo usamos la clase therian que habiamos definido antes como clase base o superclase.
    La notacion es la siguiente:
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
    esto significa que no se necesita repetir codigo que ya esta en la clase base.
    Es decir metodos como *descansar* o *usar_habilidad* ya estan definidos en la clase base "Therian".
    Pero vemos que solo sobreescribimos el metodo *presentarse* esto es porque cada subclase puede tener su propia forma de presentarse, asi  usamos el polimorfismo.
    
    
    
    """
)



hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 