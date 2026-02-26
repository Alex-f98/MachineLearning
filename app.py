import streamlit as st

#Defino pagina home
home                        = st.Page("python_course/modules/home.py", title="Home")

#Defino paginas para el repaso de python
intro_py                    = st.Page("python_course/modules/introduction.py", title="Introducción")
list_tuple_dict             = st.Page("python_course/modules/list_tuple_dict.py", title="Listas, tuplas y diccionarios")
list_comprehension_slicing  = st.Page("python_course/modules/list_comprehension_slicing.py", title="Listas de comprensión y slicing")
numpy_matrix_op             = st.Page("python_course/modules/numpy_matrix_op.py", title="Nunpy y operaciones con matrices")
f_classes_obj                 = st.Page("python_course/modules/classes_obj.py", title="Funciones, Clases y objetos")

#Defino paginas para simulaciones
simulaciones                = st.Page("python_course/modules/template.py", title="Simulaciones")


# Agrupar en secciones (nivel 1: sección, nivel 2: páginas)
pages = {
    "General"       : [home],
    "Repaso Python" : [intro_py,
                       list_tuple_dict,
                       list_comprehension_slicing,
                       numpy_matrix_op,
                       f_classes_obj],
    "Simulaciones"  : [simulaciones]
}

# Barra de navegación (en sidebar o arriba)
pg = st.navigation(pages, position="sidebar")  # o "sidebar"
pg.run()


#st.markdown("### 👨‍🔧 Test de python, por que toy al pedo [PepeCantoralPHD](https://www.youtube.com/playlist?list=PLWzLQn_hxe6bXCy0vjTGCspt2IrDUyUYm)")

#st.image("python_course/image/estudioso.png")

#st.markdown("---")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)