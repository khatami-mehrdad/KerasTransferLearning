from logging import error
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

IMAGE_CNT_MODE = 0
PATH = ['cats_and_dogs_filtered', 'cats_and_dogs_filtered_10', 'cats_and_dogs_filtered_100']
validation_dir = os.path.join(PATH[IMAGE_CNT_MODE], 'validation')

MODEL = 'MobileNetV2'
# MODEL = 'ResNet50'


### dataset
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

class_names = validation_dataset.class_names

### Use buffered prefetching to load images
AUTOTUNE = tf.data.AUTOTUNE
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

### Use data augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

### rescale pixel values from `[0, 255]` to `[-1, 1]`
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

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

### Load saved checkpoints
saved_files={}
saved_files['MobileNetV2'] = [  'MobileNetV2_cats_and_dogs_filtered.npy',
                                'MobileNetV2_first10_cats_and_dogs_filtered.npy',
                                'MobileNetV2_first100_cats_and_dogs_filtered.npy']

saved_files['ResNet50'] = [ 'ResNet50_cats_and_dogs_filtered.npy',
                            'ResNet50_first10_cats_and_dogs_filtered.npy',
                            'ResNet50_first100_cats_and_dogs_filtered.npy']

fc_w = np.load( saved_files[MODEL][IMAGE_CNT_MODE])

### Add a classification head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
num_outputs=len(class_names)
prediction_layer = tf.keras.layers.Dense(num_outputs,
                                        use_bias=False,
                                        kernel_initializer=tf.keras.initializers.Constant(fc_w),
                                        ) # cats and dogs
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

base_learning_rate = 0.0001
multiclass_TF_paradigm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  
              metrics=['accuracy'],
              run_eagerly=True,)   



loss, accuracy = multiclass_TF_paradigm_model.evaluate(validation_dataset)
print('Validation accuracy :', accuracy)
print('Validation loss :', loss)

# ### Compile the model
# base_learning_rate = 0.0001
# multiclass_TF_paradigm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  
#               metrics=['accuracy'],
#               run_eagerly=True,)   