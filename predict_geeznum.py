from keras.models import load_model
import cv2
import numpy as np

model = load_model('myclass_num.h5')

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ima1 = cv2.imread('test/5.jpg')

ima1 = np.reshape(ima1,[1,28,28,3])

class1 = model.predict_classes(ima1)


print(class1)
