import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pandas as pd
from functools import partial

#Read the training files and testing files by panda
data_folder_train = pd.read_csv("csvTrainImages 60k x 784.csv",header=None)
data_folder_train_labels = pd.read_csv("csvTrainLabel 60k x 1.csv",header=None)
data_folder_test = pd.read_csv("csvTestImages 10k x 784.csv",header=None)
data_folder_test_labels=pd.read_csv("csvTestLabel 10k x 1.csv",header=None)


# Convert images and labels to tensors
train_images = tf.convert_to_tensor(data_folder_train, dtype=tf.float32)
train_labels = tf.convert_to_tensor(data_folder_train_labels, dtype=tf.int32)
test_images = tf.convert_to_tensor(data_folder_test, dtype=tf.float32)
test_labels = tf.convert_to_tensor(data_folder_test_labels, dtype=tf.int32)

train_images = tf.reshape(train_images, (60000, 28,28))
test_images = tf.reshape(test_images,(10000,28,28))

# Create a TensorFlow dataset from the data
batch_size=64
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# Shuffle and batch the training dataset
train_dataset = train_dataset.shuffle(buffer_size=len(train_images)).batch(batch_size)


# Choose CPU or GPU to execute the code
devices = tf.config.list_physical_devices("GPU")
if len(devices) > 0:
    device = "GPU"
else:
    device = "CPU"

print(f"Using {device} device")

# Convolution Neural Network
DefaultConv2D=partial(tf.keras.layers.Conv2D, kernel_size=3, activation = 'relu', padding= 'SAME')
'''partial() function to define a thin wrapper
around the Conv2D class, called DefaultConv2D: it simply avoids having to repeat
the same weight values over and over again'''
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu', input_shape= (28,28,1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10))
model.summary()


# Choose the optimizer algorithm, loss function, and evaluation metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], #the model will compute the accuracy using SparseCategoricalAccuracy()
)

# Training process
num_epochs=5
model.fit(
    train_dataset,
    epochs=num_epochs,
    batch_size=batch_size,
)

#Testing process
test_dataset_batched = test_dataset.batch(batch_size)
test_loss, test_accuracy = model.evaluate(test_dataset_batched)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

print("Done!")


#60000 samples/64 (batch size) = 938 batches