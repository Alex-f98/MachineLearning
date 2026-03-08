import streamlit as st

# Update the layout to make better use of screen width
st.set_page_config(layout='wide')

#Defino pagina home
home                        = st.Page("python_course/pages/home.py", title="Home")

#Defino paginas para el repaso de python
path_python = "python_course/pages/intro_python"
intro_py                    = st.Page(f"{path_python}/introduction.py", title="Introducción")
list_tuple_dict             = st.Page(f"{path_python}/list_tuple_dict.py", title="Listas, tuplas y diccionarios")
list_comprehension_slicing  = st.Page(f"{path_python}/list_comprehension_slicing.py", title="Listas de comprensión y slicing")
numpy_matrix_op             = st.Page(f"{path_python}/numpy_matrix_op.py", title="Nunpy y operaciones con matrices")
f_classes_obj               = st.Page(f"{path_python}/classes_obj.py", title="Funciones, Clases y objetos")

#Defino paginas para simulaciones
path_simulation = "python_course/pages/simulation"
grad_desc                   = st.Page(f"{path_simulation}/grad_desc.py", title="Gradiente descendente- en desarrollo")
regularization_reg          = st.Page(f"{path_simulation}/regularization_reg.py", title="Regularización Ridge")
ml_viz                      = st.Page("python_course/pages/simulation/ml_viz.py", title="Visualización de ML")

#Conformal prediction.
cp_intro                    = st.Page("python_course/pages/cp/intro_cp.py", title="Introducción")

# Agrupar en secciones (nivel 1: sección, nivel 2: páginas)
pages = {
    "General"       : [home],
    "Repaso Python" : [intro_py,
                       list_tuple_dict,
                       list_comprehension_slicing,
                       numpy_matrix_op,
                       f_classes_obj],
    "Simulaciones"  : [grad_desc, regularization_reg, ml_viz]#,
    #"Conformal Predictión" : [cp_intro]
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