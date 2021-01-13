import streamlit as st
import os

def banner():
    st.image('./images/banner3.jpg', use_column_width=True)

def copyright():
    st.markdown("© 2020 and beyond  - Copyright reserved to the respective authors")

def getBiblio(current_path):
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
    path = current_path + os.sep + 'test'
    from lib import tools
    files = tools.listdirectory(path)
    files = [os.path.split(f)[1] for f in files]
    return path, files

def initControls(biblio_path, biblio_files, show_slider=True):
    if show_slider:
        seuil_confiance = st.slider(label="Seuil de confiance de l'identification", min_value=70., max_value=100., value=90., step=1.)
    else:
        seuil_confiance = 0
    local_options = {"Dans la bibliothèque" : False, "Sur un disque" : True}
    localisation = st.radio("Choisir un fichier image :", list(local_options.keys()))
    buffer = None
    filename = ''
    if local_options[localisation]:
        buffer = st.file_uploader('Télécharger une image (png ou jpg)', type=['png', 'jpg', 'jpeg'])
    else:
        filename = st.selectbox("Fichier de la bibliothèque", biblio_files)
        filename = biblio_path + os.sep + filename
        buffer = open(filename, "rb")
    return buffer, seuil_confiance
