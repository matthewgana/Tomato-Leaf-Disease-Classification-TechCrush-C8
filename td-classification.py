import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import matplotlib.pyplot as plt
import numpy as np

print("--- Initializing Deep Learning Model for MNIST ---")

# Step 1: Load the dataset
# MNIST is built directly into Keras, so we don't need external CSVs
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Step 2: Preprocess the data
# Pixel values range from 0 to 255. Neural networks learn better when data is scaled.
# We normalize the data to sit strictly between 0 and 1.
X_train = X_train / 255.0
X_test = X_test / 255.0

# Step 3: Build the Model Architecture
model = Sequential([
    # Input Layer: Flatten the 28x28 2D image array into a 1D array of 784 pixels
    Flatten(input_shape=(28, 28)),

    # Hidden Layer: 128 nodes with ReLU activation
    Dense(128, activation='relu'),

    # Regularization: Randomly drop 20% of nodes during training to prevent overfitting
    Dropout(0.2),

    # Output Layer: 10 nodes (one for each possible digit 0-9)
    # Softmax outputs a probability distribution across all 10 classes
    Dense(10, activation='softmax')
])

# Step 4: Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the Model
print("\n--- Starting Training Phase ---")
# We train for 5 epochs (5 passes through the entire training dataset)
history = model.fit(X_train, y_train, epochs=5, validation_split=0.1)

# Step 6: Evaluate on Unseen Test Data
print("\n--- Evaluating Model on Test Data ---")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# Step 7: Make Predictions and Visualize Results
print("\n--- Visualizing Predictions ---")
predictions = model.predict(X_test)

# Plot the first 5 images in the test set alongside their predicted labels
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i], cmap='gray')

    # np.argmax gets the index of the highest probability in the output array
    predicted_label = np.argmax(predictions[i])
    actual_label = y_test[i]

    # Color the title green if correct, red if wrong
    color = 'green' if predicted_label == actual_label else 'red'
    plt.title(f"Pred: {predicted_label}\nTrue: {actual_label}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()