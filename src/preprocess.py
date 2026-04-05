from tensorflow.keras.preprocessing.image import ImageDataGenerator
from image_utils import apply_gaussian_blur

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def create_train_generator(train_dir):
    """Create the training data generator with rescaling and Gaussian blur."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        preprocessing_function=apply_gaussian_blur,
    )

    print("\nLoading Training Data:")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )
    return train_generator


def create_val_generator(val_dir):
    """Create the validation data generator with rescaling and Gaussian blur."""
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        preprocessing_function=apply_gaussian_blur,
    )

    print("\nLoading Validation Data:")
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    return val_generator


def create_test_generator(test_dir):
    """Create the test data generator with rescaling and Gaussian blur."""
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        preprocessing_function=apply_gaussian_blur,
    )

    print("\nLoading Test Data:")
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    return test_generator
