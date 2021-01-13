import streamlit as st
import os
import tensorflow as tf
import global_lib as gl
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


cur_path = os.path.abspath(os.path.realpath(os.getcwd()))
#model_path = cur_path + os.sep + 'models' + os.sep + 'unet' + os.sep + 'saved_model.h5'
model_path = cur_path + os.sep + 'models' + os.sep + 'unet' + os.sep + 'weights.h5'
biblio_path, biblio_files = gl.getBiblio(cur_path)

@st.cache(allow_output_mutation=True)
def unet(pretrained_weights = None, input_size = (256,256,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=2)(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2))(conv4)

    conv5 = Conv2D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(32, (3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    ### Compilation du modèle avec la fonction de perte 'loss dice' et la métrique 'accuracy'
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-3), loss=LossDice,
                  metrics=['accuracy'])
    
    ### Si le modèle a précédemment été entrainé, les poids sauvegardés sont chargés
    if (pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model

@st.cache(allow_output_mutation=True)
def initModel(saved_path=''):
    segmenter = None
    if os.path.isfile(saved_path):
        with st.spinner(text="Chargement du réseau de neurones ..."):
            segmenter = unet(input_size=(256,256,3), pretrained_weights=saved_path)
            # segmenter = tf.keras.models.load_model(saved_path)
    else:
        st.error('Impossible de charger le modèle, fichier ' + saved_path + ' non trouvé.')
    return segmenter

### Définition d'une fonction de perte : Coefficient de Dice
def LossDice(y_true, y_pred):
  numerateur  =tf.reduce_sum(y_true*y_pred, axis=(1, 2))
  denominateur=tf.reduce_sum(y_true+y_pred, axis=(1, 2))
  dice=2*numerateur/(denominateur+1E-4)
  return 1-dice

def PILToCV(buffer):
    pil_image = Image.open(buffer).convert('RGB') 
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    return open_cv_image[:, :, ::-1].copy()

def CVToTensor(cvImage):
    img_rgb = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
    img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32)
    return tf.image.resize(img_tensor, size=(256,256))

def bounding_box(image, mask):
    bounding_box_color=(255, 0, 0)
    countour_surface=10000
    mask = mask.squeeze()
    mask = cv2.merge([mask,mask,mask])
    mask = mask * 256
    mask = mask.astype(np.uint8)
    img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(src=img, thresh=127, maxval=255, type=0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > countour_surface:
            x,y,w,h = cv2.boundingRect(contour)
            image = cv2.rectangle(img=image, pt1=(x, y), pt2=(x+w, y+h), color=bounding_box_color, thickness=2)
    cv2.destroyAllWindows()
    return image

def segmentation(buffer, model):
    with st.spinner(text="Calcul de la segmentation ..."):
        im = PILToCV(buffer)
        mask = CVToTensor(im)
        mask = tf.reshape(mask,(1,256,256,3))
        pred = model.predict(mask)

        fig = plt.figure(figsize=(20,10))
        plt.subplot(231)
        plt.imshow(im)
        plt.title("Image d'origine")
        plt.axis('off')

        plt.subplot(232)
        plt.imshow(pred.reshape(256,256),cmap='gray')
        plt.title("Masque généré par le modèle")
        plt.axis('off')

        # superposition du masque et de l'image
        plt.subplot(233)
        plt.imshow(im)
        plt.imshow(pred.reshape(256,256),alpha=0.5)
        plt.title("Superposition")
        plt.axis('off');

        # Zone de commentaire
        plt.subplot(234)
        plt.text(0, 0.9,'Application de la segmentation', fontsize=20)
        plt.text(0, 0.8,'=> Image segmentée', fontsize=20)
        plt.text(0, 0.5,'Calcul du contour délimitant la feuille', fontsize=20)
        plt.text(0, 0.4,'=> Image délimitée', fontsize=20)
        plt.axis('off')

        # La segmentation est réalisée en multipliant le masque et l'image à segmenter
        seg=(im*pred.reshape(256,256,1)).astype(int)
        plt.subplot(235)
        plt.imshow(seg)
        plt.title("Image segmentée")
        plt.axis('off')

        # BoundedBox
        bb_im = bounding_box(im, pred)
        plt.subplot(236)
        plt.imshow(bb_im)
        plt.title("Image délimitée")
        plt.axis('off')

        st.pyplot(fig)

def main():
    segmenter = initModel(model_path)

    gl.banner()
    
    st.markdown('# Segmentation')
    st.markdown("Sélectionnez un fichier image pour procéder à sa segmentation.")
    img_file_buffer, seuil = gl.initControls(biblio_path, biblio_files, False)
    # Si l'utilisateur a sélectionné une image
    if img_file_buffer is not None:
        segmentation(img_file_buffer, segmenter)    

    gl.copyright()