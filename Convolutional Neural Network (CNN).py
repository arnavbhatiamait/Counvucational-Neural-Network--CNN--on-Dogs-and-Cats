# %% [markdown]
# Convolutional Neural Network (CNN)

# %% [markdown]
# Importing the libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# %%
tf.__version__

# %% [markdown]
# Part 1 Data Preprocessing

# %% [markdown]
# Preprocessing The Training set

# %%
train_data_generator=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
# rescale is feature scaling
# shear_range is transformation
training_set=train_data_generator.flow_from_directory("dataset/dataset/training_set",target_size=(64,64),batch_size=32,class_mode='binary')
# class mode will be categorical for more than 1 output

# %% [markdown]
# Preprocessing The Testing set

# %%
# ! the images of the test set should have the same intact for testing
test_data_generator=ImageDataGenerator(rescale=1./255)
testing_set=test_data_generator.flow_from_directory("dataset/dataset/test_set",target_size=(64,64),batch_size=32,class_mode='binary')

# %% [markdown]
# Part 2 Building the CNN

# %% [markdown]
# Initilization of the CNN

# %%
cnn=tf.keras.models.Sequential()

# %% [markdown]
# Step 1 - Convolution

# %%
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu",input_shape=[64,64,3]))

# %% [markdown]
# Step 2 - Pooling

# %%
# ! max pooling
cnn.add(tf.keras.layers.MaxPool2D(strides=2,pool_size=2))

# %% [markdown]
# Adding Second Convolutional Layer

# %%
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(strides=2,pool_size=2))


# %% [markdown]
# Step 3 Flattening

# %%
cnn.add(tf.keras.layers.Flatten())

# %% [markdown]
# Step 4 Fully connected layer

# %%
# ? same as ANN
cnn.add(tf.keras.layers.Dense(activation="relu",units=128))
# !can take same as ann ie 6 but better with 128

# %% [markdown]
# Step 5 Output Layer

# %%
cnn.add(tf.keras.layers.Dense(activation="sigmoid",units=1))

# %% [markdown]
# Part 3 Training The CNN

# %% [markdown]
# Compiling the CNN

# %%
cnn.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

# %% [markdown]
# Training the CNN on the training set and evaluating it on the Test set

# %%
cnn.fit(x=training_set,validation_data=testing_set,epochs=25,)

# %% [markdown]
# Part 4 Making a single prediction

# %%
from tensorflow.keras.preprocessing import image
test_img=image.load_img("dataset/dataset/check/WhatsApp Image 2024-04-21 at 15.47.17_3170494c.jpg",target_size=(64,64))
test_img=image.img_to_array(test_img)
test_img=np.expand_dims(test_img,axis=0)
# making the right format
result=cnn.predict(test_img)
training_set.class_indices
if result[0][0]==1:
    prediction="Dog"
else: 
    prediction="Cat"

# %%
print(prediction)

# %% [markdown]
# making predictions

# %%
def pred(a):
    test_img=image.load_img(a,target_size=(64,64))
    test_img=image.img_to_array(test_img)
    test_img=np.expand_dims(test_img,axis=0)
    # making the right format
    result=cnn.predict(test_img)
    training_set.class_indices
    if result[0][0]==1:
        return "Dog"
    else: 
        return"Cat"

# %%
print(pred("dataset/dataset/check/WhatsApp Image 2024-04-21 at 15.47.17_60251636.jpg"))

# %%
print(pred("dataset/dataset/test_set/dogs/dog.4988.jpg"))

# %%
print(pred("dataset/dataset/test_set/cats/cat.4026.jpg"))


