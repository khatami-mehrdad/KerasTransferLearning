from logging import error
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

IMAGE_CNT_MODE = 1
PATH = 'cats_and_dogs_filtered'
TRAIN_PATH = ['train', 'train_first10', 'train_first100']
train_dir = os.path.join(PATH, TRAIN_PATH[IMAGE_CNT_MODE])
validation_dir = os.path.join(PATH, 'validation')

# MODEL = 'MobileNetV2'
MODEL = 'ResNet50'

epochs=20
base_learning_rate = 0.0001


### dataset
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

class_names = train_dataset.class_names

### Use buffered prefetching to load images
AUTOTUNE = tf.data.AUTOTUNE
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

### Use data augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

### rescale pixel values from `[0, 255]` to `[-1, 1]`

### Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
print('IMG_SHAPE = ', IMG_SHAPE)
if MODEL=='MobileNetV2':
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
elif MODEL=='ResNet50':
    preprocess_input = tf.keras.applications.resnet50.preprocess_input
    base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
else:
    error("Model Not supported")

### feature extractor converts each `160x160x3` image into a `5x5x1280` block of features
image_batch, label_batch = next(iter(validation_dataset))
feature_batch = base_model(image_batch)
print('feature_batch.shape = ', feature_batch.shape)\

### Freeze the convolutional base
base_model.trainable = False

### Add a classification head
num_outputs=len(class_names)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(num_outputs,) # cats and dogs
final_output_layer=tf.keras.layers.Softmax()
#
inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = prediction_layer(x)    # adds the nodes
outputs = final_output_layer(x)  # change2: adds softmax
multiclass_TF_paradigm_model = tf.keras.Model(inputs, outputs)   # this is now the learning network

multiclass_TF_paradigm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  
              metrics=['accuracy'],
              run_eagerly=True,)   

history = multiclass_TF_paradigm_model.fit(train_dataset,
                    epochs=epochs,
                    validation_data=validation_dataset)


loss, accuracy = multiclass_TF_paradigm_model.evaluate(validation_dataset)
print('Validation accuracy :', accuracy)
print('Validation loss :', loss)

### Learning curves

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()