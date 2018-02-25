import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from keras.applications import VGG16, VGG19, Xception, ResNet50, InceptionV3, InceptionResNetV2, MobileNet
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img

# Global parameters
image_size = 256
train_folder = "Images/train/"
test_folder = "Images/test/"
train_batchsize = 64
val_batchsize = 16
batch_size = 100


def rotate_spatial_images(folders):
    for folder in folders:
        files = os.listdir(folders + '/' + folder)
        for img in files:
            img_file = cv2.imread(folders + '/' + folder + '/' + img)
            img_file_90 = np.rot90(img_file)
            img_file_180 = np.rot90(img_file_90)
            img_file_270 = np.rot90(img_file_180)
            cv2.imwrite(folders + '/' + folder + '/' + img[:-4] + '_90.tif', img_file_90)
            cv2.imwrite(folders + '/' + folder + '/' + img[:-4] + '_180.tif', img_file_180)
            cv2.imwrite(folders + '/' + folder + '/' + img[:-4] + '_270.tif', img_file_270)


def create_model_with_additional_layers(model_name, image_size):
    if model_name == "vgg16":
        pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_name == "vgg19":
        pretrained_model = VGG19(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_name == "xception":
        pretrained_model = Xception(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_name == "resnet50":
        pretrained_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_name == "inceptionV3":
        pretrained_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_name == "inceptionresnetV2":
        pretrained_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_name == "mobilenet":
        pretrained_model = MobileNet(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))
    else:
        pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    # Freeze all the layers
    for layer in pretrained_model.layers[:]:
        layer.trainable = False
    # Check the trainable status of the individual layers
    for layer in pretrained_model.layers:
        print(layer, layer.trainable)
    # Create the model
    model = models.Sequential()
    # Add the vgg convolutional base model
    model.add(pretrained_model)
    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(21, activation='softmax'))
    # Show a summary of the model. Check the number of trainable parameters
    print(model.summary())
    return model


def create_model_with_retrainable_layers(model_name, image_size):
    if model_name == "vgg16":
        pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_name == "vgg19":
        pretrained_model = VGG19(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_name == "xception":
        pretrained_model = Xception(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_name == "resnet50":
        pretrained_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    elif model_name == "inceptionV3":
        pretrained_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    else:
        pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

    # Freeze all the layers
    for layer in pretrained_model.layers[:-4]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in pretrained_model.layers:
        print(layer, layer.trainable)

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(pretrained_model)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(21, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    print(model.summary())

    return model


def train_model(model, train_folder, test_folder, train_batchsize, val_batchsize, image_size, filename,
                epochs = 3,classmode='categorical', lr=1e-4):
    # No Data augmentation
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Data Generator for Training data
    train_generator = train_datagen.flow_from_directory(
            train_folder,
            target_size=(image_size, image_size),
            batch_size=train_batchsize,
            class_mode=classmode)

    # Data Generator for Validation data
    validation_generator = validation_datagen.flow_from_directory(
            test_folder,
            target_size=(image_size, image_size),
            batch_size=val_batchsize,
            class_mode=classmode,
            shuffle=False)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['acc'])

    # Train the Model
    history = model.fit_generator(
      train_generator, train_generator.n // train_batchsize, epochs=epochs, workers=4,
        validation_data=validation_generator, validation_steps=validation_generator.n // val_batchsize)

    # Save the Model
    model.save(filename)

    return model, history


def plot_training_process(history):
    # Plot the accuracy and loss curves
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

from random import randint

def show_result(model, test_folder, image_size, classmode='categorical'):
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # Create a generator for prediction
    validation_generator = validation_datagen.flow_from_directory(
        test_folder,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

    # Get the filenames from the generator
    fnames = validation_generator.filenames

    # Get the ground truth from generator
    ground_truth = validation_generator.classes

    # Get the label to class mapping from the generator
    label2index = validation_generator.class_indices

    # Getting the mapping from class index to class label
    idx2label = dict((v, k) for k, v in label2index.items())

    # Get the predictions from the model using the generator
    predictions = model.predict_generator(validation_generator,
                                          steps=validation_generator.samples / validation_generator.batch_size,
                                          verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    errors = np.where(predicted_classes != ground_truth)[0]
    print("No of errors = {}/{}".format(len(errors), validation_generator.samples))

    # Show the errors
    for i in range(5):
        ind = randint(0, len(errors) - 1)
        pred_class = np.argmax(predictions[errors[ind]])
        pred_label = idx2label[pred_class]

        title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
            fnames[errors[i]].split('/')[0],
            pred_label,
            predictions[errors[i]][pred_class])

        original = load_img('{}/{}'.format(test_folder, fnames[errors[i]]))
        plt.figure(figsize=[7, 7])
        plt.axis('off')
        plt.title(title)
        plt.imshow(original)
        plt.show()

if __name__ == "__main__":
    # Try VGG 16
    model_vgg16 = create_model_with_additional_layers(model_name="vgg16", image_size=image_size)
    model_trained_vgg16, history_vgg16 = train_model(epochs=20, lr=1e-4, model=model_vgg16, train_folder=train_folder, test_folder=test_folder, train_batchsize=train_batchsize, val_batchsize=val_batchsize, image_size=image_size)
    plot_training_process(history=history_vgg16)
    show_result(model_trained_vgg16, test_folder=test_folder, image_size=image_size)

    # Try VGG 19
    model_vgg19 = create_model_with_additional_layers(model_name="vgg19", image_size=image_size)
    model_trained_vgg19, history_vgg19 = train_model(model=model_vgg19, train_folder=train_folder, test_folder=test_folder,
            train_batchsize=train_batchsize, val_batchsize=val_batchsize, image_size=image_size, filename="vgg19.h5",
                                         lr=1e-5, epochs=20)
    plot_training_process(history=history_vgg19)
    show_result(model_trained_vgg19, test_folder=test_folder, image_size=image_size)

    model_xception = create_model_with_additional_layers(model_name="xception", image_size=image_size)
    model_trained_xception, history_xception = train_model(model=model_xception, train_folder=train_folder, test_folder=test_folder,
        train_batchsize=train_batchsize, val_batchsize=val_batchsize, image_size=image_size, filename="xception.h5",
                                         lr=1e-5, epochs=5)
    plot_training_process(history=history_xception)
    show_result(model_trained_xception, test_folder=test_folder, image_size=image_size)

    model_resnet50 = create_model_with_additional_layers('resnet50', image_size=image_size)
    model_trained_resnet50, history_resnet50 = train_model(model=model_resnet50, train_folder=train_folder, test_folder=test_folder,
        train_batchsize=train_batchsize, val_batchsize=val_batchsize, image_size=image_size, filename="xception.h5",
                                         lr=3e-5, epochs=20)
    plot_training_process(history=history_resnet50)
    show_result(model_trained_resnet50, test_folder=test_folder, image_size=image_size)

    model_inceptionV3 = create_model_with_additional_layers('inceptionV3', image_size=image_size)
    model_trained_inceptionV3, history_inceptionV3 = train_model(model=model_inceptionV3, train_folder=train_folder,
                                            test_folder=test_folder, train_batchsize=train_batchsize, val_batchsize=val_batchsize,
                                            image_size=image_size, filename="inceptionV3.h5", lr=1e-5, epochs=20)
    plot_training_process(history=history_inceptionV3)
    show_result(model=model_trained_inceptionV3, test_folder=test_folder, image_size=image_size)

    model_vgg16_retrainable_layers = create_model_with_retrainable_layers("vgg16", image_size=image_size)
    model_trained_vgg16_retrainable_layers, history_vgg16_retrainable_layers = train_model(epochs=5, lr=1e-4, model=model_vgg16_retrainable_layers, train_folder=train_folder,
            test_folder=test_folder, train_batchsize=train_batchsize, val_batchsize=val_batchsize, image_size=image_size,
                                                     filename="vgg16_lats4.h5")
    plot_training_process(history_vgg16_retrainable_layers)
    show_result(model=model_trained_vgg16_retrainable_layers, image_size=image_size, test_folder=test_folder)

    model_vgg19_retrainable_layers = create_model_with_retrainable_layers("vgg19", image_size=image_size)
    model_trained_vgg19_retrainable_layers, history_vgg19_retrainable_layers = train_model(epochs=20, lr=1e-5, model=model_vgg19_retrainable_layers, train_folder=train_folder,
            test_folder=test_folder, train_batchsize=train_batchsize, val_batchsize=val_batchsize, image_size=image_size,
                                                     filename="vgg19_lats4.h5")
    plot_training_process(history_vgg19_retrainable_layers)
    show_result(model=model_trained_vgg19_retrainable_layers, image_size=image_size, test_folder=test_folder)

    model_xception_retrainable_layers = create_model_with_retrainable_layers("xception", image_size=image_size)
    model_trained_xception_retrainable_layers, history_xception_retrainable_layers = train_model(epochs=20, lr=5e-5, model=model_xception_retrainable_layers, train_folder=train_folder,
            test_folder=test_folder, train_batchsize=train_batchsize, val_batchsize=val_batchsize, image_size=image_size,
                                                     filename="xception_lats4.h5")
    plot_training_process(history_xception_retrainable_layers)
    show_result(model=model_trained_xception_retrainable_layers, image_size=image_size, test_folder=test_folder)

    model_resnet50_retrainable_layers = create_model_with_retrainable_layers("resnet50", image_size=image_size)
    model_trained_resnet50_retrainable_layers, history_resnet50_retrainable_layers = train_model(epochs=20, lr=1e-4, model=model_resnet50_retrainable_layers, train_folder=train_folder,
             test_folder=test_folder, train_batchsize=train_batchsize, val_batchsize=val_batchsize, image_size=image_size,
                                                      filename="resnet50_lats4.h5")
    plot_training_process(history_resnet50_retrainable_layers)
    show_result(model=model_trained_resnet50_retrainable_layers, image_size=image_size, test_folder=test_folder)

    model_inceptionV3_retrainable_layers = create_model_with_retrainable_layers('inceptionV3', image_size=image_size)
    model_trained_inceptionV3_retrainable_layers, history_inceptionV3_retrainable_layers = train_model(epochs=20, lr=1e-5, model=model_inceptionV3_retrainable_layers, train_folder=train_folder,
            test_folder=test_folder, train_batchsize=train_batchsize, val_batchsize=val_batchsize, image_size=image_size,
                                                     filename="inceptionV3_lats4.h5")
    plot_training_process(history_inceptionV3_retrainable_layers)
    show_result(model=model_trained_inceptionV3_retrainable_layers, image_size=image_size, test_folder=test_folder)



