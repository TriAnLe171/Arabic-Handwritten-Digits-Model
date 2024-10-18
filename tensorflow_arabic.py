import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pandas as pd

#Read the training files and testing files by panda
data_folder_train = pd.read_csv("D:\mlr\Arabic Handwritten Digits Dataset CSV\csvTrainImages 60k x 784.csv",header=None)
data_folder_train_labels = pd.read_csv("D:\mlr\Arabic Handwritten Digits Dataset CSV\csvTrainLabel 60k x 1.csv",header=None)
data_folder_test = pd.read_csv("D:\mlr\Arabic Handwritten Digits Dataset CSV\csvTestImages 10k x 784.csv",header=None)
data_folder_test_labels=pd.read_csv("D:\mlr\Arabic Handwritten Digits Dataset CSV\csvTestLabel 10k x 1.csv",header=None)


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

# Neural network 
# Don't need to do the low-level details, keras can do it
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(512, activation='sigmoid'),
  tf.keras.layers.Dense(256, activation='sigmoid'),
  tf.keras.layers.Dense(10)
])

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