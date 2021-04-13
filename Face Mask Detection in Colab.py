import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, Dense, Flatten, BatchNormalization,MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from google.colab import files

upload = files.upload()
!unzip data.zip
!rm data.zip

batch_size = 8
epochs = 30

direct = 'data'
img_datagen=ImageDataGenerator(validation_split= 0.2,
                               rescale= 1./255,
                               rotation_range = 40,
                               width_shift_range= 0.2,
                               height_shift_range=0.2,
                               zoom_range= 0.2,
                               horizontal_flip= True,
                               fill_mode= 'nearest')
train_generator = img_datagen.flow_from_directory(direct,
                                                target_size = (70, 70),
                                                batch_size = batch_size,
                                                color_mode = "rgb",
                                                class_mode = 'binary',
                                                shuffle= True,
                                                seed = 42,
                                                subset = 'training' )

valid_generator = img_datagen.flow_from_directory(direct,
                                                target_size = (70,70),
                                                batch_size = batch_size,
                                                color_mode = "rgb",                                            
                                                shuffle= True,
                                                seed = 42,
                                                subset ='validation')

#generate a batch of images and labels from the training set
imgs, labels = next(train_generator)

#plotting function

def plotImages(images_arr):
    fig, axes = plt.subplots(1, batch_size, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
#displaying the images and thier labels where as 0 with mask and 1 without mask
plotImages(imgs);
print(labels);

model = Sequential([
                    Conv2D(filters=32, kernel_size=(3,3),activation='relu',padding='same',input_shape=(70,70,3)),
                    MaxPool2D(pool_size=(2,2), strides=2),
                    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding= 'same'),
                    MaxPool2D(pool_size=(2,2), strides =2),
                    Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding= 'same'),
                    MaxPool2D(pool_size=(2,2), strides =2),
                    Flatten(),
                    Dense(units=64, activation= 'relu'),
                    #means the output is 0,1 (the labels) and the P(c=0) +P(c=1) = 1 
                    Dense(units=1, activation='sigmoid'), 

])
model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
#Plotting the loss of validation and training 
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochstoplot = range(1,epochs+1)
plt.plot(epochstoplot, loss_train, 'g', label='Training loss')
plt.plot(epochstoplot, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Plotting the accuracy of validation and training 
accur_train = history.history['accuracy']
accur_val = history.history['val_accuracy']
plt.plot(epochstoplot, accur_train, 'g', label='Training accuracy')
plt.plot(epochstoplot, accur_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

import numpy as np  

from IPython.display import Image, display
TGREEN =  '\033[1;37;42m'
TRED =    '\033[1;37;41m'
for i in range (1,17):
  img_directory = str(i) + '.jpg'
  img_pred= image.load_img('/content/test-70x70/'+img_directory,target_size = (70,70))
  img_pred = image.img_to_array(img_pred)
  img_pred = np.expand_dims(img_pred, axis = 0)

  prediction = model.predict(img_pred)
  display(Image(img_directory,width= 150, height=150))
  print("\n")
  if(int(prediction[0][0]) == 0):
    print(TGREEN + "The person is wearing a mask. \n")
  else:
    print(TRED + "The person is not wearing a mask.\n")