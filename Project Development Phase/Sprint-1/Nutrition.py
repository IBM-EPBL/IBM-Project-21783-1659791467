seconds_in_a_day = 24 * 60 * 60
seconds_in_a_day

seconds_in_a_week = 7 * seconds_in_a_day
seconds_in_a_week

import numpy as np
from matplotlib import pyplot as plt

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)

plt.title("Sample Visualization")
plt.show()

from google.colab import drive
drive.mount('/content/drive')

cd/content/drive/MyDrive/Nutrition Image Analysis using CNN and Rapid API/Dataset/TRAIN_SET

ls/content/drive/MyDrive/Nutrition Image Analysis using CNN and Rapid API/Dataset/TEST_SET/APPLES/n07740461_10011.jpg

pwd/content/drive/MyDrive/Nutrition Image Analysis using CNN and Rapid API/Dataset/TEST_SET/APPLES/n07740461_10080.jpg


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,zoom_range=0.2,horizontal_flip=True,vertical_flip=False)

test_datagen=ImageDataGenerator(rescale=1./255)

ls

pwd

x_train=train_datagen.flow_from_directory(r"/content/drive/.shortcut-targets-by-id/1zpnSFRUQNazuPj95mSAIz0dLj-Ekk8AG/Nutrition Image Analysis using CNN and Rapid API/Dataset/TRAIN_SET",target_size=(64,64),class_mode='categorical',batch_size=24)

x_test=test_datagen.flow_from_directory(r"/content/drive/.shortcut-targets-by-id/1zpnSFRUQNazuPj95mSAIz0dLj-Ekk8AG/Nutrition Image Analysis using CNN and Rapid API/Dataset/TEST_SET",target_size=(64,64),class_mode='categorical',batch_size=24)

x_train.class_indices


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Convolution2D,MaxPooling2D,Flatten

model=Sequential()

model.add(Convolution2D(64,(3,3),input_shape=(64,64,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.summary()

32*(3*3*3+1)

model.add(Dense(300,input_dim=4,activation='relu'))

model.add(Dense(150,activation='relu'))

model.add(Dense(5,activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

len(x_train)

4118/24

len(x_test)

929/24

model.fit_generator(x_train,steps_per_epoch=len(x_train),validation_data=x_test,validation_steps=len(x_test),epochs=10)

ls

model.save('fruit.h5')

ls

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model=load_model('fruit.h5')

img=image.load_img(r"/content/drive/MyDrive/Nutrition Image Analysis using CNN and Rapid API/Dataset/TEST_SET/APPLES/n07740461_10211.jpg")

img

img=image.load_img(r"/content/drive/MyDrive/istockphoto-898671450-170667a.jpg",target_size=(64,64))

x=image.img_to_array(img)

x

x=np.expand_dims(x,axis=0)

prediction = model.predict(x)
index=['APPLES',  'BANANA' , 'ORANGE', 'PINEAPPLE', 'WATERMELON']
prediction

y=np.argmax(prediction)

index[y]