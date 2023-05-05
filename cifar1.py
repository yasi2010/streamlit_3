import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
st.title('تشخیص 10 کلاسه')
st.write('')
st.write('.این برنامه قادر به تشخیص 10 کلاس متفاوت شامل هواپیما، اتوموبیل، پرنده، گربه، گوزن، سگ، قورباغه اسب، کشتی و کامیون است')
st.write('')
file = st.file_uploader(label=':بارگذاری تصویر', type=['jpg','jpeg','png'], accept_multiple_files=False, key=None)
IMAGE_SIZE = 32
num_classes = 10


model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
model1.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model1.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model1.add(tf.keras.layers.MaxPool2D())
model1.add(tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
model1.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))


model1.load_weights('model.h5')


if file is not None:
    image = Image.open(file)
    st.image(image)
    image = np.array(image)
    # image = image[:,:,::-1].copy()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    # image = 255 - image.astype(float)
    # image /= 255
    # plt.imshow(image)
    # st.image(image)
    images = image.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)
    predictions1 = model1.predict(images)
    print(predictions1)
    predictions1 = np.argmax(predictions1, axis=1)
    print(predictions1)
    labels = ['هواپیما', 'اتومبیل', 'پرنده', 'گربه','گوزن','سگ','قورباغه','اسب','کشتی','کامیون']
    st.write(':تصویر بارگذاری شده')
    st.title(labels[predictions1[0]])
    
    st.write('')
    st.write('')
    st.write('')
    st.write('این وب اپلیکیشن توسط یاسی توسعه یافته است.')

