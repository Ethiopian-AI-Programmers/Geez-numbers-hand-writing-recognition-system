from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Activation
import cv2
import numpy as np

classifier = Sequential()

classifier.add(Convolution2D(32,(3,3),padding='same', input_shape = (28, 28, 3), activation = 'relu', data_format='channels_last'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Convolution2D(64,(3,3),padding='same', activation = 'relu', data_format='channels_last'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(4, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set  = train_datagen.flow_from_directory('am_numbers',
                                                target_size=(28,28),
                                                batch_size=50,
                                                class_mode='categorical')





classifier.fit_generator(training_set,
                        epochs = 10,
                        steps_per_epoch=400
                        )
'''
ima = cv2.imread('nnnn.jpg')
ima = np.reshape(ima,[1,28,28,3])
class1 = classifier.predict_classes(ima)
print(class1)
'''
classifier.save('myclass_num.h5')
