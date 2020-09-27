import os
import random

import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import resize
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)

plt.style.use("ggplot")

# Set some parameters
im_width = 128
im_height = 128
border = 5
path_train = "data/TGS/input/train/"
path_test = "data/TGS/input/test/"
path_plots = "plots/TGS/"


def train_model():

    X, y = get_data(path_train, train=True)

    # Split train and valid
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=2018
    )

    input_img = Input((im_height, im_width, 1), name="img")
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    """data augmentation"""

    data_gen_args = dict(horizontal_flip=True, vertical_flip=False)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 2018
    bs = 32

    image_generator = image_datagen.flow(
        X_train, seed=seed, batch_size=bs, shuffle=True
    )
    mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=bs, shuffle=True)

    # Just zip the two generators to get a generator that provides augmented images and masks at the same time
    train_generator = zip(image_generator, mask_generator)

    """data augmentation end"""

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(
            "model-tgs.h5", verbose=1, save_best_only=True, save_weights_only=True
        ),
    ]

    results = model.fit_generator(
        train_generator,
        steps_per_epoch=len(X_train) // bs,
        epochs=100,
        callbacks=callbacks,
        validation_data=(X_valid, y_valid),
    )

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(
        np.argmin(results.history["val_loss"]),
        np.min(results.history["val_loss"]),
        marker="x",
        color="r",
        label="best model",
    )
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig(path_plots + "learning_curve.png")

    # Load best model
    model.load_weights("model-tgs.h5")

    # Evaluate on validation set (this must be equal to the best log_loss)
    model.evaluate(X_valid, y_valid, verbose=1)

    # Predict on train, val and test
    preds_train = model.predict(X_train, verbose=1)
    preds_val = model.predict(X_valid, verbose=1)

    # Threshold predictions
    # .astype(np.uint8) converts to unsigned integer (0 to 255)
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.6).astype(np.uint8)

    # Check if training data looks all right
    i = 1
    while i < 160:
        ix = random.randint(0, len(X_train))
        plot_sample(
            X_train,
            y_train,
            preds_train,
            preds_train_t,
            ix=ix,
            save_as=path_plots + "train/pred_train_" + str(ix) + ".png",
        )
        i += 1

    # Check if valid data looks all right
    i = 1
    while i < 40:
        ix = random.randint(0, len(X_valid))
        plot_sample(
            X_valid,
            y_valid,
            preds_val,
            preds_val_t,
            ix=ix,
            save_as=path_plots + "valid/pred_val_" + str(ix) + ".png",
        )
        i += 1


def get_data(path, train=True):
    # Get and resize train images and masks

    ids = next(os.walk(path + "images"))[2]  # all the files in the directory
    print("No. of images = ", len(ids))
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print("Getting and resizing images ... ")

    # for n, id_ in tqdm(enumerate(ids), total=len(ids)): # add progress bar?
    for n, id_ in enumerate(ids):

        # Load images
        img = load_img(path + "/images/" + id_, color_mode="grayscale")
        x_img = img_to_array(img)
        x_img = resize(x_img, (128, 128, 1), mode="constant", preserve_range=True)

        # Load masks
        if train:
            img = load_img(path + "/masks/" + id_, color_mode="grayscale")
            y_img = img_to_array(img)
            y_img = resize(y_img, (128, 128, 1), mode="constant", preserve_range=True)

        # Save images
        X[n, ..., 0] = normalise(x_img.squeeze())
        if train:
            y[n] = normalise(y_img)
    print("Done!")
    if train:
        return X, y
    else:
        return X


def normalise(vector):
    """ Rescale a vector to ensure all entries are in range 0->1 """
    vector_min = vector.min()
    vector_max = vector.max()
    if vector_max == vector_min:
        return vector
    return (vector - vector_min) / (vector_max - vector_min)


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
    )(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
    )(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(
        input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm
    )
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def plot_sample(X, y, preds, binary_preds, ix=None, save_as="plot_sample_rand.png"):
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap="seismic")
    # 'seismic' is a diverging colormap in Matplotlib
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors="k", levels=[0.5])
    ax[0].set_title("Seismic")

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title("Salt")

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors="k", levels=[0.5])
    ax[2].set_title("Salt Predicted")

    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors="k", levels=[0.5])
    ax[3].set_title("Salt Predicted binary")
    plt.savefig(save_as)


if __name__ == "__main__":
    train_model()
