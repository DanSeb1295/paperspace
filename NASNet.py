from keras.applications.nasnet import NASNetLarge
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD
import os
from tqdm import tqdm
import cv2
import numpy as np
from random import shuffle
import h5py

categories = ["BabyBibs", "BabyHat", "BabyPants", "BabyShirt", "PackageFart", "womanshirtsleeve", "womencasualshoes",
              "womenchiffontop", "womendollshoes", "womenknittedtop", "womenlazyshoes", "womenlongsleevetop",
              "womenpeashoes", "womenplussizedtop", "womenpointedflatshoes", "womensleevelesstop",
              "womenstripedtop", "wrapsnslings"]

sourceDir = ""

num_classes = 18

def load_images():
    if os.path.exists("Original Images_Labeled.npy"):
        origImages = np.load("Original Images_Labeled.npy")
        print(len(origImages), "Images Loaded")
        return origImages
    origImages = []
    for i in os.listdir(sourceDir):
        print("-------------------  Reading {} ------------------- ".format(i))
        if i == ".DS_Store" or i == "Thumbs.db":
            continue
        folderName = os.path.join(sourceDir, i)
        for filename in tqdm(os.listdir(folderName)):
            if filename == ".DS_Store" or filename == "Thumbs.db":
                continue
            img_path = os.path.join(folderName, filename)
            image = cv2.imread(img_path)
            if np.all(image == None):
                print(filename)
                continue
            image = cv2.resize(image, (331, 331))
            label = [0]*18
            label[categories.index(i)] = 1
            origImages.append([image, label])
    print("\nNumber of images read:", len(origImages))
    # print("-------------------  SAVING -------------------")
    # np.save("Original Images_Labeled.npy", origImages)
    return origImages

def split_data(dataset):
    print("-------------------  SPLITING DATASET -------------------")
    shuffle(dataset)
    training_Data = dataset[:30000]
    validation_Data = dataset[30000:]
    train_X = np.array([i[0] for i in training_Data])
    train_Y = np.array([i[1] for i in training_Data])
    val_X = np.array([i[0] for i in validation_Data])
    val_Y = np.array([i[1] for i in validation_Data])
    return train_X, train_Y, val_X, val_Y

def mock_images():
    train_X = np.array(np.random.rand(500, 331, 331, 3))
    train_Y = np.array(np.random.rand(500,18))
    val_X = np.array(np.random.rand(100, 331, 331, 3))
    val_Y = np.array(np.random.rand(100,18))
    return train_X, train_Y, val_X, val_Y

# dataset = load_images()
# train_X, train_Y, val_X, val_Y = split_data(dataset)
train_X, train_Y, val_X, val_Y = mock_images()

# create the base pre-trained model
print("-------------------  INITIALISING MODEL -------------------")
base_model = NASNetLarge(include_top=False, weights='imagenet')

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print("-------------------  TRAINING LAST LAYER -------------------")
# train the model on the new data for a few epochs
model.fit(train_X, train_Y, batch_size=128, epochs = 3, validation_data=(val_X,val_Y))
print("-------------------  SAVING -------------------")
model.save('NASNet.h5')

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:250]:
   layer.trainable = False
for layer in model.layers[250:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
print("-------------------  TRAINING CONV LAYER -------------------")
model.fit(train_X, train_Y, batch_size=128, epochs = 3, validation_data=(val_X,val_Y))
print("-------------------  SAVING -------------------")
model.save('NASNet.h5')
