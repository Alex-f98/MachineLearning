import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title, hide_pages

add_page_title()

show_pages(
    [   
        Page("./app.py", "Test", "💻"),

        # # 2024 Content
        Section("Repaso de Python", "🐍"),
        Page("python_course/modules/introduction.py", "Introducción", "📚", in_section=True),
        Page("python_course/modules/list_tuple_dict.py", "Listas, Tuplas y Diccionarios", "📋", in_section=True),
        Page("python_course/modules/list_comprehension_slicing.py", "List Comprehension y Slicing", "✂️", in_section=True),
        Page("python_course/modules/numpy_matrix_op.py", "Numpy, Matrices y Operaciones", "🔢", in_section=True),
        #Page("python_course/modules/variables.py", "Variables", "1️⃣", in_section=True),
        #Page("dezoomcamp/About.py", "About", icon="🖼️", in_section=False) 
    ]   
)

hide_pages(["Thank you"])

st.markdown("### 👨‍🔧 Test de python, por que toy al pedo [PepeCantoralPHD](https://www.youtube.com/playlist?list=PLWzLQn_hxe6bXCy0vjTGCspt2IrDUyUYm)")

st.image("python_course/image/estudioso.png")

st.markdown("---")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 