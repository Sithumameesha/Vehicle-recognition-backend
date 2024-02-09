import tensorflow.compat.v1 as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models

batch_size = 32
img_size = (299, 299)
epochs = 10
num_classes = 1

train_data_dir = 'vehicle/train'
validation_data_dir = 'vehicle/validation'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

optimizer = tf.train.AdamOptimizer()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)


history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

model.save('vehicle_recognition_model.h5')
