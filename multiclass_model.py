from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def define_dense_model_single_layer(input_length, activation_f='sigmoid', output_length=1):
    """
    Define a dense model with a single layer.

    Parameters:
    input_length (int): The number of inputs.
    activation_f (str): The activation function.
    output_length (int): The number of outputs (number of neurons).

    Returns:
    keras.Sequential: The defined single-layer model.
    """
    model = keras.Sequential([
        keras.layers.Dense(output_length, activation=activation_f, input_shape=(input_length,))
    ])
    return model

def define_dense_model_with_hidden_layer(input_length, 
                                         activation_func_array=['relu', 'sigmoid'],
                                         hidden_layer_size=10,
                                         output_length=1):
    """
    Define a dense model with a hidden layer.

    Parameters:
    input_length (int): The number of inputs.
    activation_func_array (list): Activation functions for the hidden and output layers.
    hidden_layer_size (int): The number of neurons in the hidden layer.
    output_length (int): The number of outputs (neurons in the output layer).

    Returns:
    keras.Sequential: The defined model with one hidden layer.
    """
    model = keras.Sequential([
        keras.layers.Dense(hidden_layer_size, activation=activation_func_array[0], input_shape=(input_length,)),
        keras.layers.Dense(output_length, activation=activation_func_array[1])
    ])
    return model



def get_mnist_data():
    """Get the MNIST data."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255 
    return (x_train, y_train), (x_test, y_test)

def fit_mnist_model(x_train, y_train, model, epochs=2, batch_size=2):
    """
    Fit the model to the data.

    Parameters:
    x_train (array): Training data.
    y_train (array): Training labels.
    model (keras.Model): The model to be trained.
    epochs (int): Number of epochs for training.
    batch_size (int): Batch size for training.

    Returns:
    keras.Model: The trained model.
    """
    # One-hot encode the labels for multiclass classification
    y_train_encoded = keras.utils.to_categorical(y_train, num_classes=10)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(x_train, y_train_encoded, epochs=epochs, batch_size=batch_size)
    
    return model
  
def evaluate_mnist_model(x_test, y_test, model):
    """
    Evaluate the model on the test data.

    Parameters:
    x_test (array): Test data.
    y_test (array): Test labels.
    model (keras.Model): The trained model.

    Returns:
    tuple: Loss and accuracy of the model on test data.
    """
    # One-hot encode the labels for multiclass classification
    y_test_encoded = keras.utils.to_categorical(y_test, num_classes=10)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test_encoded)
    
    return loss, accuracy

