from os import listdir
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
from keras.applications.inception_v3 import InceptionV3
from keras.layers import LayerNormalization, Input, Activation, Dense, GlobalMaxPool2D
from keras.models import Model
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


# Settings
seed = 1337
image_size = (180, 180)
epochs = 500
validation_split = 0.2

# Determine number of classes
num_classes = len(listdir("data/training"))

# Determine number of files a folder
batch_size = len(listdir("data/training/bulbasaur"))

# Generate batches of tensor image data with real-time data augmentation
data_gen = ImageDataGenerator(
    rescale = 1.0 / 255,
    horizontal_flip = True,
    vertical_flip = False,
    brightness_range = (0.5, 1.6),
    rotation_range = 11,
    validation_split = 0.17
)

# Training dataset
train_gen = image_dataset_from_directory(
    "data/training",
    labels = "inferred",
    label_mode = "int",
    class_names = None,
    color_mode = "rgb",
    batch_size = batch_size,
    image_size = image_size,
    shuffle = True,
    seed = seed,
    validation_split = validation_split,
    subset = "training",
    interpolation = "bilinear"
)

# Validation dataset
validataion_gen = image_dataset_from_directory(
    "data/validation",
    labels = "inferred",
    label_mode = "int",
    class_names = None,
    color_mode = "rgb",
    batch_size = batch_size,
    image_size = image_size,
    shuffle = True,
    seed = seed,
    validation_split = validation_split,
    subset = "validation",
    interpolation = "bilinear"
)

# Create model
def make_model(input_shape, num_classes):

    # Defining input tensor for base model
    inputs = Input(shape = input_shape)

    # Using InceptionV3 for base model because good for thousands of classes
    base_model = InceptionV3(
        include_top = False,
        weights = "imagenet",
        input_tensor = inputs,
        input_shape = None,
        pooling = None
    )

    # Entry block
    layer = base_model.layers[-1].output
    layer = LayerNormalization(
        axis = -1,
        epsilon = 0.001,
        center = True,
        scale = True,
        beta_initializer = "zeros",
        gamma_initializer = "ones",
    )(layer)
    layer = GlobalMaxPool2D()(layer)
    layer = Activation("relu")(layer)

    outputs = Dense(num_classes, activation = "softmax")(layer)
    return Model(inputs, outputs)

# Create a model with checkpoints in case system crashes
model = make_model(input_shape = image_size + (3,), num_classes = num_classes)

# Create callbacks
callbacks = [
    ModelCheckpoint("models/checkpoint_{epoch:02d}_{val_loss:.2f}.h5"),
    EarlyStopping(monitor = "val_loss'", patience = 24, verbose = 1),
    TensorBoard("./logs")
]

# A LearningRateSchedule that uses an exponential decay schedule
learning_rate = ExponentialDecay(initial_learning_rate = 1e-4, decay_steps = 20000, decay_rate = 0.5, staircase = True)

# Compiling the model
model.compile(
    optimizer = Adam(learning_rate = learning_rate),
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"],
)

# Train the model
model.fit(
    train_gen,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validataion_gen,
)

# Save as a single master model
model.save("PokeRecognitionInceptionV3.h5")
