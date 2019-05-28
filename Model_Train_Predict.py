""""

Notes:

weight map functionality has been removed (24/09/18): accuracy does not improve with usage of weigth map


number of conv layers: 24
number of relu units: 23
number of sigmoid units: 1 (after last conv layer)
number of batch norm layers: 10
number of max pooling layers: 4


"""


import os
import cv2
import re
import numpy as np
import math
import copy
import glob
from math import sqrt
from skimage.morphology import remove_small_objects
from scipy.ndimage import label



class GPU_or_CPU(object):


    def __init__(self, mode):

        self.mode = mode

    def ret_mode(self):

        if self.mode == "GPU":
            print("Train / Predict on GPU")

        elif self.mode == "CPU":
            print("Train / Predict on CPU")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        return self.mode


from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal as gauss
from keras import backend as K
from keras import losses
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.preprocessing.image import array_to_img

from Training_DataGenerator import *


"""

16/01/18

Image size needs to be divisible by 2:
To allow seamless tiling of the output segmentation map, input tile size needs to be selected such that all 2x2
max pooling operations are applied to a layer with an even x and y size

"""


class MitoSegNet(object):


    def __init__(self, path, img_rows, img_cols, org_img_rows, org_img_cols):

        self.path = path

        self.img_rows = img_rows
        self.img_cols = img_cols

        self.org_img_rows = org_img_rows
        self.org_img_cols = org_img_cols


    def natural_keys(self, text):

        # sorting of alphanumerical strings

        def atoi(text):
            return int(text) if text.isdigit() else text

        return [atoi(c) for c in re.split('(\d+)', text)]

    def load_data(self, wmap, vbal):


        print('-' * 30)
        print('Load train images...')
        print('-' * 30)

        imgs_train = np.load(self.path + os.sep + "npydata" + os.sep +"imgs_train.npy")
        imgs_mask_train = np.load(self.path + os.sep + "npydata" + os.sep + "imgs_mask_train.npy")

        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')

        imgs_train /= 255
        imgs_mask_train /= 255

        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0

        if wmap == True:

            imgs_weights = np.load(self.path + os.sep + "npydata" + os.sep + "imgs_weights.npy")
            imgs_weights = imgs_weights.astype('float32')

            # setting background pixel weights to vbal (because of class imbalance)
            imgs_weights[imgs_weights == 1] = vbal

            return imgs_train, imgs_mask_train, imgs_weights

        else:

            return imgs_train, imgs_mask_train


    def get_mitosegnet(self, wmap, lr):

        inputs = Input(shape=(self.img_rows, self.img_cols, 1))
        print(inputs.get_shape(), type(inputs))


        # core mitosegnet (modified u-net) architecture
        ######################################
        ######################################

        """

        as of 10/10/2018

        Contracting path:

            5 sections each consisting of the following layers:

                convolution 
                batchnorm 
                activation 
                convolution
                batchnorm
                activation 
                pooling

        Expanding path: 

            4 sections each consisting of the following layers:

                inverse convolution (upsampling)
                merging
                convolution
                convolution 

            1 section containing only one convolutional layer 

        """


        # batchnorm architecture (batchnorm before activation)
        ######################################################################

        conv1 = Conv2D(64, 3, padding='same', kernel_initializer=gauss())(inputs)
        print("conv1 shape:", conv1.shape)
        batch1 = BatchNormalization()(conv1)
        act1 = Activation("relu")(batch1)

        conv1 = Conv2D(64, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(act1)  # conv1
        print("conv1 shape:", conv1.shape)
        batch1 = BatchNormalization()(conv1)
        act1 = Activation("relu")(batch1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(act1)
        print("pool1 shape:", pool1.shape)
        ########

        ########
        conv2 = Conv2D(128, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(pool1)
        print("conv2 shape:", conv2.shape)
        batch2 = BatchNormalization()(conv2)
        act2 = Activation("relu")(batch2)

        conv2 = Conv2D(128, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(act2)  # conv2
        print("conv2 shape:", conv2.shape)
        batch2 = BatchNormalization()(conv2)
        act2 = Activation("relu")(batch2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(act2)
        print("pool2 shape:", pool2.shape)
        ########

        ########
        conv3 = Conv2D(256, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(pool2)
        print("conv3 shape:", conv3.shape)
        batch3 = BatchNormalization()(conv3)
        act3 = Activation("relu")(batch3)

        conv3 = Conv2D(256, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(act3)  # conv3
        print("conv3 shape:", conv3.shape)
        batch3 = BatchNormalization()(conv3)
        act3 = Activation("relu")(batch3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(act3)
        print("pool3 shape:", pool3.shape)
        ########

        ########
        conv4 = Conv2D(512, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(pool3)
        batch4 = BatchNormalization()(conv4)
        act4 = Activation("relu")(batch4)

        conv4 = Conv2D(512, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(act4)  # conv4
        batch4 = BatchNormalization()(conv4)
        act4 = Activation("relu")(batch4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(act4)
        ########

        ########
        conv5 = Conv2D(1024, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(pool4)
        batch5 = BatchNormalization()(conv5)
        act5 = Activation("relu")(batch5)

        conv5 = Conv2D(1024, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 1024))))(act5)  # conv5
        batch5 = BatchNormalization()(conv5)
        act5 = Activation("relu")(batch5)
        ########

        up6 = Conv2D(512, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 1024))))(UpSampling2D(size=(2, 2))(act5))

        merge6 = concatenate([conv4, up6], axis=3)

        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(conv6)


        up7 = Conv2D(256, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(UpSampling2D(size=(2, 2))(conv6))

        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(conv7)


        up8 = Conv2D(128, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(UpSampling2D(size=(2, 2))(conv7))

        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(conv8)


        up9 = Conv2D(64, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(UpSampling2D(size=(2, 2))(conv8))

        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(conv9)

        conv9 = Conv2D(2, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(conv9)

        ######################################################################

        conv10 = Conv2D(1, 1, activation='sigmoid', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 2))))(conv9)


        if wmap == False:
            input = inputs
            loss = self.pixelwise_crossentropy()
        else:
            weights = Input(shape=(self.img_rows, self.img_cols, 1))
            input = [inputs, weights]

            loss = self.weighted_pixelwise_crossentropy(input[1])


        model = Model(inputs=input, outputs=conv10)

        # normally set to 1e-4
        model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=['accuracy', self.dice_coefficient])

        return model


    def dice_coefficient(self, y_true, y_pred):

        smooth = 1

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)

        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        return dice

    def pixelwise_crossentropy(self):

        def loss(y_true, y_pred):

            return losses.binary_crossentropy(y_true, y_pred)

        return loss

    def weighted_pixelwise_crossentropy(self, wmap):

        def loss(y_true, y_pred):

            return losses.binary_crossentropy(y_true, y_pred) * wmap

        return loss


    def train(self, epochs, learning_rate, batch_size, wmap, vbal, model_name):

        if ".hdf5" in model_name:
            pass
        else:
            model_name = model_name + ".hdf5"

        print("Loading data")

        if wmap == False:
            imgs_train, imgs_mask_train = self.load_data(wmap=wmap, vbal=vbal)
        else:
            imgs_train, imgs_mask_train, img_weights = self.load_data(wmap=wmap, vbal=vbal)

        print("Loading data done")

        model = self.get_mitosegnet(wmap, learning_rate)
        print("Got mitosegnet")


        print(self.path + os.sep + model_name)


        if os.path.isfile(self.path + os.sep + model_name):

            model.load_weights(self.path + os.sep + model_name)
            print("Loading weights")

        else:
            print("No previously optimized weights were loaded. Proceeding without")


        # Set network weights saving mode.
        # save previously established network weights (saving model after every epoch)

        print('Fitting model...')

        model_name_csv = model_name.split(".")[0]
        csv_logger = CSVLogger(self.path + os.sep + model_name_csv + '_training_log.csv')

        # Set callback functions to early stop training and save the best model so far
        callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                     ModelCheckpoint(filepath=self.path + os.sep +model_name, monitor='val_loss', verbose=1, save_best_only=True),
                     csv_logger]


        if wmap == True:
            x = [imgs_train, img_weights]
        else:
            x = imgs_train

        model.fit(x=x, y=imgs_mask_train, batch_size=batch_size, epochs=epochs, verbose=1,
                            validation_split=0.2, shuffle=True, callbacks=callbacks)


        info_file = open(self.path + os.sep + model_name + "_train_info.txt", "w")
        info_file.write("Learning rate: " + str(learning_rate)+
                        "\nBatch size: " + str(batch_size))
        info_file.close()

        K.clear_session()


    def predict(self, test_path, wmap, tile_size, n_tiles, model_name, pretrain):

        """
        :return:
        """

        K.clear_session()

        org_img_rows = self.org_img_rows
        org_img_cols = self.org_img_cols

        natural_keys = self.natural_keys


        def create_test_data(tile_size, n_tiles):




            # adding all image data to one numpy array file (npy)
            # all original image files are added to imgs_test.npy


            i = 0
            print('-' * 30)
            print('Creating test images...')
            print('-' * 30)

            imgs = glob.glob(test_path + os.sep + "*")


            # added 05/12/18 to avoid underscores causing problems when stitching images back together
            #if any("_" in s for s in imgs):

            for img in imgs:

                if "_" in img and ".tif" in img:

                    # split file path without filename
                    img_edited = img.split(os.sep)[:-1]

                    # join list back to path string
                    img_edited_path = os.sep.join(img_edited)

                    img_name = img.split(os.sep)[-1]
                    img_name = img_name.replace("_", "-")

                    os.rename(img, img_edited_path + os.sep + img_name)

            imgs = glob.glob(test_path + os.sep + "*")
            imgs.sort(key=natural_keys)

            # create list of images that correspond to arrays in npy file
            ################
            mod_imgs = []
            for x in imgs:

                part = x.split(os.sep)

                c = 0
                while c <= n_tiles - 1:

                    temp_str = os.sep.join(part[:-1])

                    if ".tif" in part[-1]:

                        mod_imgs.append(temp_str + os.sep + str(c) + "_" + part[-1])

                    c += 1

            ################

            imgdatas = np.ndarray((len(imgs) * n_tiles, tile_size, tile_size, 1), dtype=np.uint8)

            for imgname in imgs:

                if ".tif" in imgname:

                    img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
                    cop_img = copy.copy(img)

                    # insert split into 4 sub-images here
                    ################################################### (09/11/18)

                    y, x = img.shape

                    def small_img_size(y, x, img):

                        # todo: currently setting border only around bottom and right side in order not to move
                        # the tile within the frame (maybe it would be necessary to set a frame around entire image)

                        """
                        Takes image smaller than tile size and adds a border on the bottom and right side to expand
                        the size to 1030 x 1300

                        :param y:
                        :param x:
                        :param img:
                        :return:
                        """

                        org_height, org_width = 1030, 1300

                        #top_bottom = int(org_height - y)
                        #left_right = int(org_width - x)

                        top_bottom = int(org_height - y/2)
                        left_right = int(org_width - x/2)

                        #new_img = cv2.copyMakeBorder(img, top_bottom, top_bottom, left_right, left_right, cv2.BORDER_REFLECT)

                        # top, bottom, left, right
                        new_img = cv2.copyMakeBorder(img, 0, top_bottom, 0, left_right, cv2.BORDER_REFLECT)

                        img = new_img
                        cop_img = copy.copy(new_img)

                        y, x = img.shape

                        return img, cop_img, y, x


                    # if the input image size is smaller than the predefined tile size run this function
                    if y < tile_size or x < tile_size:

                        img, cop_img, y, x = small_img_size(y,x, img)


                    start_y = 0
                    start_x = 0

                    end_y = tile_size
                    end_x = tile_size

                    column = 0
                    row = 0

                    for n in range(n_tiles):

                        start_x, end_x, start_y, end_y, column, row = preproc.find_tile_pos(x, y, tile_size, start_x, end_x,
                                                                                         start_y, end_y, column, row)

                        img_tile = cop_img[start_y:end_y, start_x:end_x]

                        img = img_tile.reshape((tile_size, tile_size, 1))

                        imgdatas[i] = img

                        i += 1

                np.save(test_path + os.sep + 'imgs_array.npy', imgdatas)

            return mod_imgs, y, x


        def load_test_data():

            print('-' * 30)
            print('Load test images...')
            print('-' * 30)

            imgs_test = np.load(test_path + os.sep + "imgs_array.npy")
            imgs_test = imgs_test.astype('float32')

            imgs_test /= 255

            return imgs_test


        preproc = Preprocess()

        l_imgs, y, x = create_test_data(int(tile_size), int(n_tiles))
        imgs_test = load_test_data()

        # predict if no npy array exists yet
        if not os.path.isfile(test_path + os.sep + "imgs_mask_array.npy"):

            lr = 1e-4

            model = self.get_mitosegnet(wmap, lr)

            if pretrain == "":
                model.load_weights(self.path + os.sep + model_name)
            else:
                model.load_weights(pretrain)

            print('Predict test data')

            # batch size of 14 was too large to fit into GPU memory
            # verbose: show ongoing progress (0 do not show)
            imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

            np.save(test_path + os.sep + 'imgs_mask_array.npy', imgs_mask_test)

        else:

            print("\nFound imgs_mask_array.npy. Skipping prediction and converting array to images\n")

        org_img_list = []

        for img in l_imgs:

            # split file path without filename

            if ".tif" in img:

                img_name = img.split(os.sep)[-1]
                img_name = img_name.split("_")[1]

                org_img_list.append(img_name)

        org_img_list = list(set(org_img_list))
        org_img_list.sort(key=self.natural_keys)

        # saving arrays as images
        print("Array to image")

        imgs = np.load(test_path + os.sep + 'imgs_mask_array.npy')

        #####
        imgs = imgs.astype('float32')
        #####

        start_y = 0
        start_x = 0
        end_y = tile_size
        end_x = tile_size

        column = 0
        row = 0

        img_nr = 0
        org_img_list_index = 0

        temp_l = []


        small_size = False
        for n, image in zip(range(imgs.shape[0]), l_imgs):

            if img_nr == 0:

                if org_img_rows < tile_size or org_img_cols < tile_size:

                    small_size = True

                    small_y, small_x = org_img_rows, org_img_cols

                    org_img_rows, org_img_cols = y, x

                current_img = np.zeros((org_img_rows, org_img_cols))

            img = imgs[n]

            img = array_to_img(img)

            start_x, end_x, start_y, end_y, column, row = preproc.find_tile_pos(org_img_cols, org_img_rows, tile_size,
                                                                   start_x, end_x, start_y, end_y, column, row)

            current_img[start_y:end_y, start_x:end_x] = img

            current_img = current_img.astype(np.float32)
            temp_l.append(current_img)

            if small_size == True:
                current_img = np.zeros((y, x))

            else:
                current_img = np.zeros((org_img_rows, org_img_cols))

            img_nr += 1

            # once one image has been fully stitched, remove any objects below 10 px size and save
            if img_nr == n_tiles:

                start_y = 0
                start_x = 0

                end_y = tile_size
                end_x = tile_size

                column = 0
                row = 0

                #"""

                background = temp_l[0]


                if len(temp_l) > 1:

                    for tmp_n in temp_l[1:]:

                        overlay = tmp_n

                        # todo
                        """
                        this method currently does not take the number of overlaps into account, by which the added
                        intensities should be divided to get a mean value. this would require to automatically generate
                        the coordinates of all overlapping regions 
                        """

                        new_img = cv2.addWeighted(background, 1, overlay, 1, 0)
                        background = new_img

                else:

                    new_img = background

                temp_l = []

                new_img[new_img >= 128] = 255
                new_img[new_img < 128] = 0

                new_img = new_img.astype(np.uint8)

                #"""

                label_image, num_features = label(new_img)
                new_image = remove_small_objects(label_image, 10)

                new_image[new_image != 0] = 255

                # todo
                if small_size == True:

                    new_image = new_image[:small_y, :small_x]

                cv2.imwrite(test_path + os.sep + "Prediction" + os.sep + org_img_list[org_img_list_index], new_image)


                org_img_list_index+=1
                img_nr = 0

        K.clear_session()



