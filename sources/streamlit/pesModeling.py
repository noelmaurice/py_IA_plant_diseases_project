import streamlit as st
import global_lib as gl

def paragrapheModele1():
    st.markdown("#### Architecture")
    st.markdown("Nous sommes partis sur la structure d'un modèle U-Net.")
    st.image('./images/unet.png', use_column_width=True)
    st.markdown("#### Entraînement")
    st.markdown("Nous avons utilisé le jeu de données **plantvillage-dataset** qui contient déjà des images segmentées.")
    st.markdown("""L'entraînement s’est fait avec un unique batch de près de 34754 images de dimension 256 x 256 et 3 epochs.  
    Le modèle a été entrainé par itération avec sauvegarde régulière des poids et ajustement du taux d'apprentissage au cours
    de l'entraînement.""")
    st.warning("""
    Contrairement au modèle d'identification qui suit, par manque de temps, nous n'avons pas cherché à atteindre le meilleur 
    taux de performance possible, ni à entraîner le modèle sur d'autres images que celle du dataset. Le modèle n'est donc
    pas capable de détourer les photos ne représentant pas une feuille de plante.     
    """)
    st.markdown("#### Evaluation")
    st.markdown("La performance approche les 95% de prédictions correctes.")

def paragrapheModele2():
    st.markdown("### Première itération")
    st.markdown("Nous avons d'abord construit un réseau de neurones à convolution (CNN) sur mesure.")
    st.markdown("#### Architecture")
    st.markdown("""
Le modèle disposait de :
  * 2 couches de convolution (Conv2D) pour extraire les caractéristiques importantes
  * 2 couches de pooling (MaxPooling2D) pour réduire la taille des images
  * 1 couche d'applatissement (Flatten) pour applatir les matrices en vecteurs
  * 3 couches denses (Dense) pour l'apprentissage
  * 2 couches d'abandon (Dropout) pour réduire la quantité de paramètres à entrainer
""")
    display_source1 = st.checkbox("Afficher le code source", key='source1')
    if display_source1:
        code = """
        classifier = Sequential()
        classifier.add(Conv2D(filters = 64, kernel_size = (5, 5), strides = 1, input_shape = (image_size, image_size, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Conv2D(filters = 128, kernel_size = (3, 3), strides = 1, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(units = 160, activation = 'relu'))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(units = 96, activation = 'relu'))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(units = 38, activation = 'softmax'))
        classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        """
        st.code(code)
    st.markdown("#### Entraînement")
    st.markdown("""Il s’est fait avec un batch_size de 64 et 220 epochs avec des images de dimension 128 x 128.  
    Le modèle a été entrainé par itération avec sauvegarde progressive et régulière des poids des couches denses.""")
    st.markdown("#### Evaluation")
    col1, col2 = st.beta_columns([1, 1])
    with col1:
        st.text("Courbe de perte")
        st.image('./images/loss_model1.png', use_column_width=True)
    with col2:
        st.text("Courbe de performance")
        st.image('./images/perf_model1.png', use_column_width=True)


def paragrapheModele3():
    st.markdown("### Seconde itération")
    st.markdown("#### Architecture")
    st.markdown("Notre second modèle s'appuie sur un modèle VGG16 entrainé sur les données **imagenet** que nous avons complété.")
    st.image('./images/architecture_vgg16.png', use_column_width=True)
    st.markdown("""
Nous avons complété le modèle avec les couches suivantes :
  * 1 couche de pooling
  * 3 couches denses
  * 2 couches d'abandon
""")
    display_source2 = st.checkbox("Afficher le code source", key='source2')
    if display_source2:
        code = """
        base_model = VGG16(weights='imagenet', include_top=False) 
        for layer in base_model.layers: 
            layer.trainable = False
        model = Sequential()
        model.add(base_model) # Ajout du modèle VGG16
        model.add(GlobalAveragePooling2D()) 
        model.add(Dense(1024,activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(39, activation='softmax'))
        """
        st.code(code)
    st.markdown("#### Entraînement")
    st.markdown("""Le modèle a été entrainé sur 27 itérations sur des images de dimension 256 x 256.""")
    st.markdown("#### Evaluation")
    st.markdown("  * Performance et perte durant l'entraînement :")
    st.image('./images/model3.png', use_column_width=True)
    st.markdown("  * Focus sur les 10 derniers epochs :")
    st.image('./images/model3_10epochs.png', use_column_width=True)
    st.markdown("On constate la stagnation de la courbe sur le jeu de validation, le modèle ne s'améliore plus.")

def main():
    gl.banner()

    st.markdown('# Modélisation')
    st.markdown("## Modèle de segmentation")
    paragrapheModele1()
    st.markdown("## Modèle d'identification")
    paragrapheModele2()
    paragrapheModele3()

    gl.copyright()

    