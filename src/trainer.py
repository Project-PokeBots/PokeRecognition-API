import os
import tensorflow as tf

# Check if training folders equal validation folders
num_classes = len(next(os.walk("data/training"))[1])

# Setting variables
seed = 1337
image_size = (180, 180)
batch_size = 32
buffer_size = 32

# Training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split = 0.2,
    subset = "training",
    seed = seed,
    image_size = image_size,
    batch_size = batch_size,
)

# Validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split = 0.2,
    subset = "validation",
    seed = seed,
    image_size = image_size,
    batch_size = batch_size,
)

# Artifically produce image diversity
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
    ]
)

# Configure dataset for preformance
train_ds = train_ds.prefetch(buffer_size = buffer_size)
val_ds = val_ds.prefetch(buffer_size = buffer_size)

# Create model
def make_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape = input_shape)

    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)
    x = tf.keras.layers.Conv2D(32, 3, strides = 2, padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(64, 3, padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [128, 256, 512, 728]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding = "same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding = "same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides = 2, padding = "same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(size, 1, strides = 2, padding = "same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = tf.keras.layers.SeparableConv2D(1024, 3, padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation = "softmax")(x)
    return tf.keras.Model(inputs, outputs)


# Create a model with checkpoints in case system crashes
model = make_model(input_shape = image_size + (3,), num_classes = num_classes)

epochs = 500

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("checkpoint_{epoch}.h5"),
]

model.compile(
    optimizer = tf.keras.optimizers.Adam(1e-3),
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"],
)

model.fit(
    train_ds,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = val_ds,
)

# Save as a single model
model.save("keras_model.h5")
