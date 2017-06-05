# Imports relevant modules
import csv
import cv2
import numpy as np
import ntpath
import sklearn
import random
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.utils import shuffle
import matplotlib.image as mpimg

#Function to brighten images
def brighten_image(image):
    """Randomly brighten images for better data processing"""
    brighten_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    brighten_image[:,:,2] = brighten_image[:,:,2] * random_bright
    brighten_image = cv2.cvtColor(brighten_image,cv2.COLOR_HSV2RGB)
    
    return brighten_image

#Function to flip images
def flip_image(image,measurement):
    """Flip images and their corresponding measurements"""
    flipped = np.fliplr(image)
    measurement = - measurement
    
    return flipped, measurement


#Loading data

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

del(samples[0])

#Splitting images to training and validation samples

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

correction_factor = 0.25

def generator(samples, batch_size=258):
    """Randomly selects sample images and modify images before passing it"""
    num_samples = len(samples)
    while 1: 
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            
            for batch_sample in batch_samples:
                
                image_source = batch_sample[camera]
                file_name = ntpath.basename(image_source)
                image_path = 'data/IMG/'+ file_name
                
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                
                #Randomly picks images from left, center or right cameras 
                
                camera = random.choice([0,0,0,1,2])
                
                if camera == 0:
                    measurement = float(batch_sample[3])

                if camera == 1:
                    measurement = float(batch_sample[3]) + correction_factor

                if camera == 2: 
                    measurement = float(batch_sample[3]) - correction_factor
                    
                #Randomly augment images through brightening or flipping
                augmentation = random.choice(['brighten','brighten','flip'])

                if augmentation  == 'brighten':
                    image = brighten_image(image)


                if augmentation == 'flip':
                    image, measurement = flip_image(image, measurement)
                    
                images.append(image)
                measurements.append(measurement)
                    
            X_train = np.array(images)
            y_train = np.array(measurements)
            
            yield sklearn.utils.shuffle(X_train, y_train)

#Importing keras modules
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout, MaxPooling2D, Reshape
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam

#Model Architecture
#Input Layer
model = Sequential()
model.add(Lambda(lambda x:(x/127.5)-1., input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65,25),(0,0))))

#Layer 1
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))

#Layer 2
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))

#Layer 3
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='elu'))
model.add(MaxPooling2D([2,2]))

#Layer 4
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dropout(0.5))

#Layer 5
model.add(Dense(50))
model.add(Activation('elu'))

#Layer 6
model.add(Dense(10))
model.add(Activation('elu'))

#Layer Output
model.add(Dense(1))
          
#Training and validation
train_generator = generator(train_samples, batch_size=500)
validation_generator = generator(validation_samples, batch_size=500)
          
model.compile(loss='mse',optimizer=Adam(0.001))
model.fit_generator(train_generator, samples_per_epoch=10000, validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')


