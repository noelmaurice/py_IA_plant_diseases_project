import streamlit as st
from PIL import Image
import numpy as np
import os
import json
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.applications.vgg16 import preprocess_input, VGG16
from urllib.request import urlopen
from bs4 import BeautifulSoup, Tag
import global_lib as gl

# Initialisation des variables
cnn_name = 'cnn_vgg16_augmented'
image_size = 256
epochs = 0
n_class = 39
cur_path = os.path.abspath(os.path.realpath(os.getcwd()))
weights_path = cur_path + os.sep + 'models' + os.sep + 'vgg16' + os.sep + cnn_name + '_' + str(image_size) + '_weights'
history_path = cur_path + os.sep + 'models' + os.sep + 'vgg16' + os.sep + cnn_name + '_' + str(image_size) + '_history'
biblio_path, biblio_files = gl.getBiblio(cur_path)
url_root = 'https://plantvillage.psu.edu/topics/'
url_suffix = '/infos'

# Chargement des paramètres
with open('./data/class_params.json', 'r') as f_in:
    plant_disease_dict = json.load(f_in)
class_dict = {value['Indice']:key for (key,value) in plant_disease_dict.items()}

@st.cache(allow_output_mutation=True)
def initModel(is_weights=False):
    with st.spinner(text="Instanciation du réseau de neurones ..."):
        base_model = VGG16(weights='imagenet', include_top=False) 
        classifier = Sequential()
        classifier.add(base_model) # Ajout du modèle VGG16
        classifier.add(GlobalAveragePooling2D()) 
        classifier.add(Dense(1024,activation='relu'))
        classifier.add(Dropout(rate=0.2))
        classifier.add(Dense(512, activation='relu'))
        classifier.add(Dropout(rate=0.2))
        classifier.add(Dense(n_class, activation='softmax'))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        if is_weights:
            classifier.load_weights(weights_path)
        else:
            st.error('Impossible de charger les poids du modèle, fichier ' + weights_path + ' non trouvé.')
    return classifier

# MDByTypeInfo permet de consommer les informations disponibles sur le site plantvillage.psu.edu
def MDByTypeInfo(className, typeInfo):
    # Fonction de recherche interne spécifique au besoin de PlantExpert
    def findBalise(tag):
        return (tag.name=='h4') and (tag.contents[0].strip()==diseaseName)    

    balise = None
    # Récupération des paramètres relatifs à la classe prédite
    infos = plant_disease_dict[className]
    # Construction de l'url d'accès à l'information sur le site plantvillage.psu.edu
    url = url_root + infos['url'] + url_suffix
    # Récupération du nom de la maladie (tel qu'il est connu sur le site)
    diseaseName = infos['tag']
    # Si une maladie a été détectée
    if diseaseName != 'healthy':
        # Afficher un titre à la section
        st.markdown('### ' + typeInfo.capitalize() + ' :')
        # Ouvrir l'url du site
        page = urlopen(url)
        # Convertir la page en éléments BeautifulSoup
        soup = BeautifulSoup(page, 'html.parser')
        # Rechercher la balise décrivant l'élément recherché (la maladie détectée par le modèle)
        balise = soup.find(findBalise)
        # Si la recherche échoue
        if balise == None:
            # Prévenir l'utilisateur que l'information n'a pas été trouvée
            st.text('Tag Not found')
        # Parcourir l'arbre DOM pour retrouver la balise contenant le type d'info recherchée (symptome, traitement, ...)
        while balise != None:
            # Si l'élement en cours est une balise HTML
            if isinstance(balise, Tag) :
                # Si la balise contient la classe recherchée (symptôme, traitement, ...)
                if ('class' in balise.attrs) and (typeInfo in balise['class']):
                    # Si l'information recherchée est le traitement
                    if typeInfo == 'management':
                        # Supprimer la balise H5 
                        # le site n'a pas toujours une structure cohérente, 
                        # l'ordre des balises n'est pas toujours identique
                        balise.h5.decompose()
                    # Afficher le contenu de la balise
                    st.markdown(balise, unsafe_allow_html=True)
                    break
            # Passer à la balise suivante
            balise = balise.next_sibling
    return balise

# Fonction d'affichage des symptômes de la maladie
def MDSymptome(className):
    return MDByTypeInfo(className, 'symptoms')

# Fonction d'affichage des traitements de la maladie
def MDTraitement(className):
    return MDByTypeInfo(className, 'management')

# Fonction d'affichage de la source des informations
def MDSource(className):
    url = url_root + plant_disease_dict[className]['url'] + url_suffix
    st.markdown("### Source :")
    st.markdown(url)
    return None

# Fonction d'affichage des informations d'identification de l'image
def MDIdentification(className, rate):
    infos = plant_disease_dict[className]
    st.markdown('### Identification :')
    st.markdown('* Classe identifiée : ' + className + '\n* Confiance : ' + str(rate) + '%\n* Plante : **' + infos['Plant'] + '**' + '\n* Maladie : **' + infos['Disease'] + '**')
    return None

def prediction(img, classifier, rate):
    # Afficher une popup indiquant que la prédiction est en cours
    with st.spinner(text="Calcul de la prédiction ..."):
        img = img.resize((image_size, image_size), Image.ANTIALIAS)
        # Convertir l'image en matrice
        image_array = np.asarray(img)
        # Ajout d'une dimension supplémentaire pour la compatibilité avec le modèle
        image_array = np.expand_dims(image_array, axis=0)
        # Preprocessing de l'image
        image_array = preprocess_input(image_array)
        #image_array = np.array([image_array]) / 255
        # Lancer la prédiction
        class_predict = classifier.predict(image_array)
    # Récupérer l'indice de la classe prédite
    indice_class_predict = np.argmax(class_predict)
    # Récupérer le libellé de la classe prédite
    pred_class_name = class_dict[indice_class_predict]
    # Récupérer le taux de confiance de la prédiction
    trust_rate = round( (class_predict[0][indice_class_predict] * 100), 2 )
    # Si le taux de confiance est supérieur au seuil de confiance fixé par l'utilisateur
    if trust_rate >= rate:
        # Afficher les informations d'identification de l'image
        MDIdentification(pred_class_name, trust_rate)
        if pred_class_name != 'Other':
            # Afficher les symtômes, les traitements et la source d'information si une maladie a été prédite
            with st.spinner(text="Collecte d'informations ..."):
                MDSymptome(pred_class_name)
                MDTraitement(pred_class_name)
                MDSource(pred_class_name)
    else: # Afficher que la prédiction a échouée
        st.markdown("L'indice de confiance de la prédiction est de " + str(trust_rate) + "%")
        st.markdown("Le seuil de confiance n'est pas atteint.")
        st.markdown("Le modèle n'a pas reconnu l'image.")
    return indice_class_predict, pred_class_name, trust_rate


def main():
    classifier = initModel(is_weights = os.path.isfile(history_path))
    
    gl.banner()    

    st.markdown('# Prédiction')
    st.markdown("Sélectionnez un fichier image puis procédez à son identification.")
    img_file_buffer, seuil = gl.initControls(biblio_path, biblio_files)
    # Si l'utilisateur a sélectionné une image
    if img_file_buffer is not None:
        # Ouvrir le fichier Image
        imageFile = Image.open(img_file_buffer)
        # Afficher l'image
        st.image(imageFile)
        # Afficher le bouton permettant de lancer la prédiction (identification de l'image)
        btn_identification = st.button('Identification')
        # Si l'utilisateur clique sur le bouton d'identification
        if btn_identification:
            prediction(imageFile, classifier, seuil)

    gl.copyright()