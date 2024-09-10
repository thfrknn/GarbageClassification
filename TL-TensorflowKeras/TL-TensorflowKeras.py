import tensorflow as tf

# VGG16 is a pre-trained convolutional neural network model.
conv_base = tf.keras.applications.VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3)
                  )

# Display the convolutional layers of the VGG16 model.
conv_base.summary()

# Set which layers are trainable and which are frozen.
# All layers until 'block5_conv1' are frozen.
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    layer.trainable = set_trainable

# Create a new empty model.
model = tf.keras.models.Sequential()

# Add the VGG16 model as a convolutional base.
model.add(conv_base)

# Flatten the output of the convolutional base to a vector.
model.add(tf.keras.layers.Flatten())

# Add a dense layer with 256 neurons and ReLU activation.
model.add(tf.keras.layers.Dense(256, activation='relu'))

# Add the final dense layer with 2 neurons and softmax activation for binary classification.
model.add(tf.keras.layers.Dense(2, activation='softmax'))

# Compile the model with binary cross-entropy loss and RMSprop optimizer.
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=['acc'])

# Display the summary of the created model.
model.summary()

# Define the directories for the training, validation, and test data.
train_dir = 'data/train'
validation_dir = 'data/val'
test_dir = 'data/test'

# Apply data augmentation techniques to the training data to prevent overfitting.
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255,  # Normalize pixel values to the range 0-1.
      rotation_range=40,  # Apply random rotations.
      width_shift_range=0.2,  # Apply horizontal shifts.
      height_shift_range=0.2,  # Apply vertical shifts.
      shear_range=0.2,  # Apply shearing transformations.
      zoom_range=0.2,  # Apply random zooms.
      horizontal_flip=True,  # Randomly flip images horizontally.
      fill_mode='nearest'  # Fill in newly created pixels.
      )

# Create a generator for the training data.
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical'
        )

# Use only normalization for validation data (no augmentation).
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
        )

# Create a generator for the validation data.
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical'
        )

# Train the model using the training and validation data.
history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=5)

# Save the trained model to a file.
model.save('trained_tf_model.h5')

# Use only normalization for test data (no augmentation).
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
        )

# Create a generator for the test data.
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical'
        )

# Evaluate the model on the test data and print the accuracy.
test_loss, test_acc = model.evaluate(test_generator, steps=50)
print('test acc:', test_acc)
