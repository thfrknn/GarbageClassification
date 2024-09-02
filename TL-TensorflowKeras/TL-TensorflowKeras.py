import tensorflow

# VGG16 is a pre-trained convolutional neural network model.
conv_base = tensorflow.keras.applications.VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3)
                  )

