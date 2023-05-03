import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
st.title('تشخیص 10 کلاسه')
st.write('')
st.write('This is a python app which classifies a brain MRI into one of the four classes : ')
st.write(' No tumor, Pituitary tumor,Meningioma tumor or Glioma tumor')
file = st.file_uploader(label='Upload image', type=['jpg','jpeg','png'], accept_multiple_files=False, key=None)
IMAGE_SIZE = 32
num_classes = 10

# from tensorflow.keras.applications import EfficientNetB0
# effnet = EfficientNetB0(weights = None,include_top=False,input_shape=(IMAGE_SIZE,IMAGE_SIZE, 3))
# model1 = effnet.output

model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
model1.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model1.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model1.add(tf.keras.layers.MaxPool2D())
model1.add(tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
model1.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

# model1.add(tf.keras.layers.Flatten())
# model1.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

# model1 = tf.keras.layers.GlobalAveragePooling2D()(model1)
# model1 = tf.keras.layers.Dropout(0.5)(model1)
# model1 = tf.keras.layers.Dense(4, activation = 'softmax')(model1)
# model1 = tf.keras.models.Model(inputs = effnet.input, outputs = model1)

model1.load_weights('model.h5')


if file is not None:
    image = Image.open(file)
    # st.image(image)
    image = np.array(image)
    # image = image[:,:,::-1].copy()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    # image = 255 - image.astype(float)
    # image /= 255
    # plt.imshow(image)
    st.image(image)
    images = image.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)
    predictions1 = model1.predict(images)
    print(predictions1)
    predictions1 = np.argmax(predictions1, axis=1)
    print(predictions1)
    labels = ['airplane', 'automobile', 'bird', 'cat','deer','dog','frog','horse','ship','truck']
    st.write('Prediction over the uploaded image:')
    st.title(labels[predictions1[0]])

