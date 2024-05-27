import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of Sigmoid activation function
def sigmoid_derivative(x):
    return x * (1 - x)


# Input dataset
inputs = np.array([[1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
                   [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                   [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0],
                   [1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
                   [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
                   [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
                   [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
                   [0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
                   [1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
                   [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                   [1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
                   [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1],
                   [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1],
                   [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1]])

# Output dataset
outputs = np.array([[0],
                    [1],
                    [0],
                    [1],
                    [0],
                    [0],
                    [1],
                    [1],
                    [0],
                    [0],
                    [0],
                    [0],
                    [1],
                    [1],
                    [1],
                    [1]])


# Seed random numbers to make calculation deterministic
randomSeed = random.randint(0, 10000)
np.random.seed(randomSeed)


# Initialize weights randomly with mean 0
synaptic_weights_1 = 2*np.random.random((16, 32)) - 1
synaptic_weights_2 = 2*np.random.random((32, 64)) - 1
synaptic_weights_3 = 2*np.random.random((64, 128)) - 1
synaptic_weights_4 = 2*np.random.random((128, 256)) - 1
synaptic_weights_5 = 2*np.random.random((256, 128)) - 1
synaptic_weights_6 = 2*np.random.random((128, 64)) - 1
synaptic_weights_7 = 2*np.random.random((64, 32)) - 1
synaptic_weights_8 = 2*np.random.random((32, 16)) - 1

# Forward propagation
layer_0 = inputs
layer_1 = sigmoid(np.dot(layer_0, synaptic_weights_1))
layer_2 = sigmoid(np.dot(layer_1, synaptic_weights_2))
layer_3 = sigmoid(np.dot(layer_2, synaptic_weights_3))
layer_4 = sigmoid(np.dot(layer_3, synaptic_weights_4))
layer_5 = sigmoid(np.dot(layer_4, synaptic_weights_5))
layer_6 = sigmoid(np.dot(layer_5, synaptic_weights_6))
layer_7 = sigmoid(np.dot(layer_6, synaptic_weights_7))
layer_8 = sigmoid(np.dot(layer_7, synaptic_weights_8))
layer_outputs = layer_8

for iteration in range(10000):

    # How much did we miss?
    layer_2_error = outputs - layer_2

    if (iteration % 1000) == 0:
        print("Error:" + str(np.mean(np.abs(layer_2_error))))

    # Multiply how much we missed by the
    # slope of the sigmoid at the values in layer_2
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)

    # How much did each layer_1 value contribute to the layer_2 error
    layer_1_error = layer_2_delta.dot(synaptic_weights_2.T)

    # Multiply how much we missed by the
    # slope of the sigmoid at the values in layer_1
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    # Update weights
    synaptic_weights_2 += layer_1.T.dot(layer_2_delta)
    synaptic_weights_1 += layer_0.T.dot(layer_1_delta)

print("Output After Training:")
print(layer_outputs)


# Define the neural network architecture
layers = [16, 32, 64, 128, 256, 128, 64, 32, 16]

# Create a TensorFlow model
model = tf.keras.models.Sequential()

# Add layers to the model
for i, layer_size in enumerate(layers):
    if i == 0:
        model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
    else:
        model.add(tf.keras.layers.Dense(layer_size, activation='relu'))

# Add a softmax output layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Create a TensorBoard callback
tensorboard_callback = TensorBoard(
    log_dir='logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(inputs, outputs, epochs=10, callbacks=[tensorboard_callback])


# Calculate the positions of the nodes
# positions = []
# for i, layer in enumerate(layers):
#     positions.append(np.array([i, layer/2]))
#
# # Create the plot
# fig, ax = plt.subplots()
#
# # Draw the nodes
# for i, layer in enumerate(layers):
#     ax.scatter(positions[i][0], positions[i][1], s=100, c='lightblue')
#     ax.text(positions[i][0], positions[i][1], f'Layer {i}\n({layer} neurons)', ha='center', va='center')
#
# # Draw the edges
# for i in range(len(layers) - 1):
#     ax.plot([positions[i][0], positions[i+1][0]], [positions[i][1], positions[i+1][1]], 'k-', lw=2)
#
# # Add axis labels and title
# ax.set_x_label('Layer')
# ax.set_y_label('Neurons')
# ax.set_title('Neural Network Architecture')
#
# # Show the plot
# plt.show()
