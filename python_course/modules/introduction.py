import streamlit as st



st.markdown("""
**Índice:**

* [General](#general)
* [Google Colab](#google-colab)
* [Importar datos al colab](#importar-datos-al-colab)
---
""")

st.markdown("### General")

#st.video("https://www.youtube.com/watch?v=AtRhA-NfS24")
            
st.info("Curso basado en el apunte de Wachenchauzer y el libro [Think Python](https://allendowney.github.io/ThinkPython/).")

st.markdown("""
Este curso de Python contiene material teorico y practico, 
intenta mostrar conceptos de forma visual a fin de entender los conceptos detras del código mediante
animaciones y explicaciones detalladas.
""", unsafe_allow_html=True)

st.info("El curso esta en desarrollo y cualquier sugerencia es bienvenida. Puede ser enviado al siguiente correo: bfuentes@fi.uba.ar")
st.markdown(
    """
Todo el material de ejemplos será probado en google colab, 
por lo que es recomendable usarlo para probar los ejemplos.

De ser posible se suministra el codigo en google colab para probar los ejemplos.

""",
unsafe_allow_html=True)


st.markdown('<a id="google-colab"></a>', unsafe_allow_html=True)
st.markdown("### Google Colab")

st.markdown("""
[Google Colaboratory (Colab)](https://colab.research.google.com/) es un entorno de desarrollo en la nube de Google que permite crear 
notebooks gratuitos para ejecutar código Python.

Este entorno proporciona una forma rápida de probar código Python sin necesidad de instalarlo en su computadora.

También, permite el acceso a GPU (limitado a 12 GB en su versión gratuita) para entrenar modelos de machine learning y ya tiene instaladas varias librerías como numpy, pandas, matplotlib, seaborn, etc.

Para usarlo hay que acceder directamente a la url de [Google Colab](https://colab.research.google.com/). o puedes crear un notebook nuevo desde el menu de archivos de
google drive.

**Opción 1 - Acceso directo:**
1. Visita [https://colab.research.google.com/](https://colab.research.google.com/)
2. Haz clic en "New notebook" o "Archivo → Nuevo notebook"
 
**Opción 2 - Desde Google Drive:**
1. Abre tu [Google Drive](https://drive.google.com/)
2. Haz clic en "Nuevo" → "Más" → "Google Colaboratory"
3. Se creará un notebook que se guardará automáticamente en tu Drive
 
""", unsafe_allow_html=True)

st.info(
    """ 
    Debes saber que Colab se ejecuta en la nube de google, por lo que si cierras el notebook sin guardarlo en google drive, perderás todo el código.
    
    No es todo malo, también te permite equivocarte y probar cosas sin preocuparte por romper nada (solo abres otro notebook y listo).
    """)

st.markdown("---")


st.markdown("""
   La interfaz básica tiene celdas de código y celdas de texto las cuales se ejecutan en orden y puedes agregar cuantas quieras.

   * Celda de codigo: Ejecuta codigo python y tambien comandos de terminal (precedidos por **!**).
   * Celda de texto: Permite escribir texto en formato markdown incluso latex.

Cada celda puede ser ejecutada individualmente o todas juntas, para ejecutarlas se puede hacerlo con el boton de play o con el atajo de teclado shift + enter.

   
   Puedes ver a la derecha la sección de recursos del notebook que estás usando en tiempo real, donde puedes ver cuánta RAM y Disco estás usando.
   Para la mayoría de los ejemplos y proyectos que necesites esto es suficiente, pero si necesitas más recursos puedes usar tu propia computadora o lo que mejor se adapte a tus necesidades.

""")
st.image("python_course/image/colabExplicacion.jpeg", caption="Colab", width=800)


st.markdown("""
    Puedes notar en la siguiente imagen que Colab en su sección de archivos (a izquierda) ahí es donde se guardarán todos los archivos creados
    y también puedes subir archivos desde tu computadora.
    Este no es mas que una carpeta principal llamada */content* y puedes ver que también tiene una carpeta llamada */sample_data* el cual contiene archivos de ejemplo para probar.

    **Todo lo creado en la carpeta /content se borrará cuando cierres la sesión de Colab** por lo que debes asegurarte de descargar los archivos generados.
""")
st.image("python_course/image/colabB1.jpeg", caption="comandos", width=800)

st.markdown("""
    También puedes inspeccionar las variables que has creado, puedes hacer clic en la zona inferior en *variables* y aparecerá una
    solapa a la izquierda que mostrará el nombre de la variable y su tipo de dato, forma y valor.

    Adicionalmente tienes más pero ya es suficiente para empezar.
""")


st.image("python_course/image/colabB2.jpeg", caption="comandos", width=800)


st.info("""Puedes notar que en la solapa izquierda tienes muchas más opciones para explotar, como buscar y reemplazar, fragmentos de código, inspección de datos,
y quizás la más interesante la de *secrets* el cual permite poner credenciales de acceso a servicios externos de manera segura como las de Hugging Face, OpenAI, Kaggle, etc.
Estas puedes verlas por tu cuenta, no son necesarias para empezar pero a medida que las necesites irás descubriendo como usarlas.
""")

st.markdown('<a id="importar-datos-al-colab"></a>', unsafe_allow_html=True)
st.markdown("### Importar datos al colab")

st.markdown("""
Para  Importar archivos al colab nos valernos de dos comandos **curl** y **Wget**.

#### curl
El comando **curl** significa "Client URL" y se usa como una forma de verificar conectividad a una URL y para transferir
datos.

El comando curl es compatible con varios protocolos, entre ellos nos interesa que es compatible con HTTP y HTTPS.

se para hacer peticiones a una url (GET, POST, PUT, DELETE, etc) y por default hace GET.

la sintaxis es la siguiente:

```
curl [OPCIONES] [URL]
```

El uso que se le dará aquí sera para descargar archivos desde una ubicación remota.

Se puede hacer de dos formas:

- -O Guardará el archivo en el directorio de trabajo actual con el mismo nombre de archivo que el remoto.
- -o permite especificar un nombre de archivo o ubicación diferente.

ej:

```
!curl -O https://upload.wikimedia.org/wikipedia/commons/6/64/Dall-e_3_%28jan_%2724%29_artificial_intelligence_icon.png
!curl -o img_ia.png https://upload.wikimedia.org/wikipedia/commons/6/64/Dall-e_3_%28jan_%2724%29_artificial_intelligence_icon.png
```

Para descargar archivos desde una ubicación remota tenemos que tener la url especifica de ese archivo.

Por ejemplo si se quiere descargar datos desde github, habrá que acceder al la ubicación del archivo y obtener la url solo del archivo.
""")
st.image("python_course/image/github_raw01.png", caption="comandos", width=800)
st.markdown("Una vez ahí habrá que acceder a la opción **raw** y copiar la url.")
st.image("python_course/image/github_raw02.png", caption="comandos", width=800)
st.markdown("Y esa url propia del archivo es la que luego se debe usar para descargar el archivo. como los ejemplos visto anteriormente")


st.markdown("#### Wget")
st.markdown("""
El comando wget significa World Wide Web Get y es un comando de terminal que se usa para descargar archivos desde una ubicación remota.

Es más directo y sirve para descargar archivos desde una ubicación remota.

Cuando se usa el comando wget, se descarga el archivo en el directorio de trabajo actual con el mismo nombre de archivo que el remoto.

Es más simple para descargas directas o grandes.

La sintaxis basica es la siguiente.

```
!wget [OPCIONES] [URL]
```

Puedes ver mas sobre las diferencias entre [wget y curl](https://medium.com/@sean.liu.job/the-differences-between-curl-and-wget-d5ad3d0b4844).

""")



hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 