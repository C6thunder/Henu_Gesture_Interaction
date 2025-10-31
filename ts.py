import tensorflow as tf
print(tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
