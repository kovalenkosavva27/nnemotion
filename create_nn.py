import numpy as np
import model
from keras.preprocessing.image import ImageDataGenerator
train_path='train'
test_path='test'
img_size = 48
batch_size = 64

datagen_train = ImageDataGenerator(horizontal_flip=True)

train_generator = datagen_train.flow_from_directory(train_path,
target_size=(img_size,img_size),
color_mode="grayscale",
batch_size=batch_size,
class_mode='categorical',
shuffle=True)

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_validation.flow_from_directory(test_path,
target_size=(img_size,img_size),
color_mode="grayscale",
batch_size=batch_size,
class_mode='categorical',
shuffle=False)

steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size
nn=model.CNN()
history = nn.fit(
x=train_generator,
steps_per_epoch=steps_per_epoch,
epochs=5,
validation_data = validation_generator,
validation_steps = validation_steps,
)
nn.save('my_model.h5')