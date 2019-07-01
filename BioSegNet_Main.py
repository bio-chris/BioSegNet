"""

Class Control

Contains all functions that are necessary for the entire program

Class Advanced Mode

Contains all functions necessary for the advanced mode

Class Easy Mode

Contains all function necessary for the easy mode

"""


from tkinter import *
import tkinter.font
import tkinter.messagebox
import tkinter.filedialog
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import webbrowser
import math
import cv2
import time
from skimage.measure import regionprops, label
from Create_Project import *
from Training_DataGenerator import *
from Model_Train_Predict import *
from Train_Val_Analyser import *


# GUI
####################

class Control():

    """
    Adds functions that can be accessed from all windows in the program
    """

    def __init__(self):
        pass

    # close currently open window
    def close_window(self, window):

        window.destroy()

    # opens link to documentation of how to use the program
    def help(self):

        webbrowser.open_new("https://github.com/bio-chris/BioSegNet")

    # go to main window
    """
    def go_back(self, current_window, root):

        current_window.quit()
        root.quit()

        try:
            root = start_window()
            root.mainloop()

        except:
            pass
    """

    # open new window with specified width and height
    def new_window(self, window, title, width, height):

        window.title(title)

        window.minsize(width=int(width/2), height=int(height/2))
        window.geometry(str(width)+"x"+str(height)+"+0+0")

    def automatic_eval_train(self, datapath):

        analyser = AnalyseData()

        aut = True

        analyser.csv_analysis(datapath, "acc", "Accuracy", aut)
        analyser.csv_analysis(datapath, "dice_coefficient", "Dice coefficient", aut)
        analyser.csv_analysis(datapath, "loss", "Loss", aut)

        plt.show()

    def eval_train(self):

        analysis_root = Tk()
        self.new_window(analysis_root, "BioSegNet Training Analysis", 450, 170)

        datapath = StringVar(analysis_root)
        popup_var = StringVar(analysis_root)

        def askopencsv():
            set_modelpath = tkinter.filedialog.askopenfilename(parent=analysis_root, title='Choose a CSV file')
            datapath.set(set_modelpath)

        #### browse for pretrained model
        text = "Select training log file"
        control_class.place_browse(askopencsv, text, datapath, 15, 20, None, None, analysis_root)
        ####

        popup_var.set("Accuracy")
        Label(analysis_root, text="Choose Metric to display", bd=1).place(bordermode=OUTSIDE, x=15, y=90)
        popupMenu_train = OptionMenu(analysis_root, popup_var, *set(["Accuracy", "Dice coefficient", "Loss"]))
        popupMenu_train.place(bordermode=OUTSIDE, x=15, y=110, height=30, width=150)

        def analysis():

            if datapath.get() != "":

                analyser = AnalyseData()

                if popup_var.get() == "Accuracy":
                    metric = "acc"
                elif popup_var.get() == "Dice coefficient":
                    metric = "dice_coefficient"
                else:
                    metric = "loss"

                aut = False

                analyser.csv_analysis(datapath.get(), metric, popup_var.get(), aut)

            else:

                tkinter.messagebox.showinfo("Error", "Entries not completed", parent=analysis_root)

        control_class.place_button(analysis_root, "Analyse", analysis, 300, 110, 30, 110)



    # adds menu to every window, which contains the above functions close_window, help and go_back
    def small_menu(self, window):

        menu = Menu(window)
        window.config(menu=menu)

        submenu = Menu(menu)
        analysis_menu = Menu(menu)
        menu.add_cascade(label="Menu", menu=submenu)
        menu.add_cascade(label="Analysis", menu=analysis_menu)

        analysis_menu.add_command(label="Evaluate training performance", command=lambda:self.eval_train())

        submenu.add_command(label="Help", command=lambda: self.help())

        # creates line to separate group items
        submenu.add_separator()

        #submenu.add_command(label="Go Back", command=lambda: self.go_back(window, root))
        submenu.add_command(label="Exit", command=lambda: self.close_window(window))

    # extract tile size, original y and x resolution, list of tile images, list of images (prior to splitting)
    def get_image_info(self, path, pretrained, multifolder):

        """
        :param path:
        :return: tile_size, y, x, tile_list, image_list
        """

        def get_shape(path, img_list):

            for img in img_list:

                if ".tif" in img:
                    img = cv2.imread(path + os.sep + img, cv2.IMREAD_GRAYSCALE)
                    y = img.shape[0]
                    x = img.shape[1]
                    break

            return y, x


        # we should not assume a new trained model should only predict images that are the same size as the ones the model
        # was trained on

        if pretrained is False:

            tiles_path = path + os.sep + "train" + os.sep + "image"
            tiles_list = os.listdir(tiles_path)

            images_path = path + os.sep + "train" + os.sep + "RawImgs" + os.sep + "image"
            images_list = os.listdir(images_path)

            tile_size_y, tile_size_x = get_shape(tiles_path, tiles_list)
            y, x = get_shape(images_path, images_list)

            return tile_size_y, y, x, tiles_list, images_list

        else:

            if multifolder is False:

                y, x = get_shape(path, os.listdir(path))

            else:

                for subfolders in os.listdir(path):
                    new_path = path + os.sep + subfolders

                    y, x = get_shape(new_path, os.listdir(new_path))
                    break

            return y, x


    def generate_project_folder(self, finetuning, project_name, path, img_datapath, label_datapath, window):

        create_project = CreateProject(project_name)

        cr_folders = False
        copy = False
        if path != "":
            cr_folders = create_project.create_folders(path=path)

            if cr_folders == True:

                if img_datapath != "" and label_datapath != "":
                    create_project.copy_data(path=path, orgpath=img_datapath, labpath=label_datapath)
                    copy = True

                else:
                    tkinter.messagebox.showinfo("Note", "You have not entered any paths", parent=window)

            else:
                pass

        else:
            tkinter.messagebox.showinfo("Note", "You have not entered any path", parent=window)

        if cr_folders == True and copy == True and finetuning == False:
            tkinter.messagebox.showinfo("Done", "Generation of project folder and copying of files successful!",
                                        parent=window)


    def get_measurements(self, imgpath, labpath, datatype, subfolder):

        if datatype != False:

            org_img_list = os.listdir(imgpath)

            img_list = [i for i in org_img_list if ".tif" in i]
            print(img_list)

            dataframe = pd.DataFrame(columns=["Image", "Measurement", "Average", "Median", "Standard Deviation",
                                              "Standard Error", "Minimum", "Maximum", "N"])

            n = 0
            for img in img_list:

                read_img = cv2.imread(imgpath + os.sep + img, -1)
                read_lab = cv2.imread(labpath + os.sep + img, cv2.IMREAD_GRAYSCALE)

                labelled_img = label(read_lab)[0]

                labelled_img_props = regionprops(label_image=labelled_img, intensity_image=read_img, coordinates='xy')

                area = [obj.area for obj in labelled_img_props]
                minor_axis_length = [obj.minor_axis_length for obj in labelled_img_props]
                major_axis_length = [obj.major_axis_length for obj in labelled_img_props]
                eccentricity = [obj.eccentricity for obj in labelled_img_props]
                perimeter = [obj.perimeter for obj in labelled_img_props]
                solidity = [obj.solidity for obj in labelled_img_props]
                mean_int = [obj.mean_intensity for obj in labelled_img_props]
                max_int = [obj.max_intensity for obj in labelled_img_props]
                min_int = [obj.min_intensity for obj in labelled_img_props]


                def add_to_dataframe(measure_str, measure, n):

                    dataframe.loc[n] = [img] + [measure_str, np.average(measure), np.median(measure), np.std(measure),
                                                np.std(measure) / np.sqrt(len(measure)), np.min(measure), np.max(measure),
                                                len(measure)]


                meas_str_l = ["Area", "Minor Axis Length", "Major Axis Length", "Eccentricity", "Perimeter", "Solidity",
                              "Mean Intensity", "Max Intensity", "Min Intensity"]
                meas_l = [area, minor_axis_length, major_axis_length, eccentricity, perimeter, solidity, mean_int, max_int,
                          min_int]

                for m_str, m in zip(meas_str_l, meas_l):

                    add_to_dataframe(m_str, m, n)
                    n+=1

            parentDirectory = os.path.abspath(os.path.join(imgpath, os.pardir))

            if subfolder == False:
                folder_name = parentDirectory.split(os.sep)[-1]
            else:
                folder_name = subfolder


            if datatype == "CSV":
                dataframe.to_csv(parentDirectory + os.sep + folder_name + "_Analysis_Table.csv")
            else:
                dataframe.to_excel(parentDirectory + os.sep + folder_name + "_Analysis_Table.xlsx")

        else:
            pass


    def prediction(self, datapath, modelpath, pretrain, model_file, batch_var, popupvar, popupvar_meas, tile_size, y, x,
                   window):

            pred_biosegnet = BioSegNet(modelpath, img_rows=tile_size, img_cols=tile_size, org_img_rows=y,
                                         org_img_cols=x)

            set_gpu_or_cpu = GPU_or_CPU(popupvar)
            set_gpu_or_cpu.ret_mode()


            if popupvar_meas == "No measurements":
                datatype = False

            elif popupvar_meas == "CSV Table":
                datatype = "CSV"

            else:
                datatype = "Excel"


            if batch_var == "One folder":

                if not os.path.lexists(datapath + os.sep + "Prediction"):
                    os.mkdir(datapath + os.sep + "Prediction")

                pred_biosegnet.predict(datapath, False, tile_size, model_file, pretrain)

                control_class.get_measurements(datapath, datapath + os.sep + "Prediction", datatype,
                                               False)

            else:

                for subfolders in os.listdir(datapath):

                    if not os.path.lexists(datapath + os.sep + subfolders + os.sep + "Prediction"):
                        os.mkdir(datapath + os.sep + subfolders + os.sep + "Prediction")

                    pred_biosegnet.predict(datapath + os.sep + subfolders, False,
                                            tile_size, model_file, pretrain)

                    labelpath = datapath + os.sep + subfolders + os.sep + "Prediction"

                    control_class.get_measurements(datapath + os.sep + subfolders, labelpath, datatype,
                                                   subfolders)

            tkinter.messagebox.showinfo("Done", "Prediction successful! Check " + datapath + os.sep +
                                        "Prediction" + " for segmentation results", parent=window)


    def place_gpu(self, popupvar, x, y, window):

        popupvar.set("GPU")
        Label(window, text="Train / Predict on", bd=1).place(bordermode=OUTSIDE, x=x, y=y)
        popupMenu_train = OptionMenu(window, popupvar, *set(["GPU", "CPU"]))
        popupMenu_train.place(bordermode=OUTSIDE, x=x+10, y=y+20, height=30, width=100)


    def place_prediction_text(self, batch_var, popupvar, popupvar_meas, window):

        control_class.place_text(window, "Apply model prediction on one folder or multiple folders?", 20, 160,
                                 None, None)

        batch_var.set("One folder")
        popupMenu_batch_pred = OptionMenu(window, batch_var, *set(["One folder", "Multiple folders"]))
        popupMenu_batch_pred.place(bordermode=OUTSIDE, x=30, y=180, height=30, width=130)

        self.place_gpu(popupvar, 20, 220, window)

        popupvar_meas.set("No measurements")
        Label(window, text="Generate measurement table", bd=1).place(bordermode=OUTSIDE, x=20, y=280)
        popupMenu_train = OptionMenu(window, popupvar_meas, *set(["No measurements", "CSV Table", "Excel Table"]))
        popupMenu_train.place(bordermode=OUTSIDE, x=30, y=300, height=30, width=130)


    def place_browse(self, func, text, text_entry,x, y, height, width, window):

        if height is None or width is None:
            Label(window, text=text, bd=1).place(bordermode=OUTSIDE, x=x, y=y)
        else:
            Label(window, text=text, bd=1).place(bordermode=OUTSIDE, x=x, y=y, height=height, width=width)

        Button(window, text="Browse", command=func).place(bordermode=OUTSIDE, x=x+370, y=y+20, height=30, width=50)

        entry = Entry(window, textvariable=text_entry)
        entry.place(bordermode=OUTSIDE, x=x+10, y=y+20, height=30, width=350)

        return entry


    def place_text(self, window, text, x, y, height, width):

        if height is None or width is None:
            Label(window, text=text, bd=1).place(bordermode=OUTSIDE, x=x, y=y)
        else:
            Label(window, text=text, bd=1).place(bordermode=OUTSIDE, x=x, y=y, height=height, width=width)

    def place_button(self, window, text, func, x, y, height, width):

        Button(window, text=text, command=func).place(bordermode=OUTSIDE, x=x, y=y, height=height, width=width)

    def place_entry(self, window, text, x, y, height, width):

        Entry(window, textvariable=text).place(bordermode=OUTSIDE, x=x, y=y, height=height, width=width)


class AdvancedMode(Control):

    """
    Contains the following functionalities, in which most parameters can be adjusted by the user:

    Start new project: create new project folder for subsequent data augmentation, training and prediction
    Continue working on existing project: augment data, train model, predict using trained model
    """

    def __init__(self):
        Control.__init__(self)

    preprocess = Preprocess()
    control_class = Control()

    # Single "Continue working on existing project" windows
    ##########################################

    # Window: Create augmented data
    def cont_data(self, old_window):

        """
        Data augmentation

        :param old_window:
        :return:
        """

        old_window.destroy()
        data_root = Tk()

        self.new_window(data_root, "BioSegNet Data Augmentation", 450, 550)
        self.small_menu(data_root)

        dir_data_path = StringVar(data_root)
        tkvar = StringVar(data_root)
        tile_size = IntVar(data_root)
        tile_number = IntVar(data_root)
        n_aug = IntVar(data_root)
        width_shift = DoubleVar(data_root)
        height_shift = DoubleVar(data_root)
        shear_range = DoubleVar(data_root)
        rotation_range = IntVar(data_root)
        zoom_range = DoubleVar(data_root)

        tkvar.set('')  # set the default option

        def askopendir():

            set_dir_data_path = tkinter.filedialog.askdirectory(parent=data_root, title='Choose a directory')
            dir_data_path.set(set_dir_data_path)

            pr_list, val_List = self.preprocess.poss_tile_sizes(set_dir_data_path + os.sep + "train" + os.sep + "RawImgs")

            if set_dir_data_path != "":

                tkvar.set(list(pr_list)[0])  # set the default option
                choices = pr_list

                popupMenu = OptionMenu(data_root, tkvar, *choices)
                popupMenu.place(bordermode=OUTSIDE, x=30, y=90, height=30, width=300)


        # on change dropdown value
        def change_dropdown(*args):

            tile_inf = tkvar.get()

            l = (tile_inf.split(" "))

            tile_size.set(int(l[3]))
            tile_number.set(int(l[-1]))


        #link function to change dropdown (tile size and number)
        tkvar.trace('w', change_dropdown)


        text= "Select BioSegNet Project directory"
        control_class.place_browse(askopendir, text, dir_data_path, 20, 10, None, None, data_root)

        control_class.place_text(data_root, "Choose the tile size and corresponding tile number", 20, 70, None, None)

        control_class.place_text(data_root, "Choose the number of augmentation operations", 20, 130, None, None)
        control_class.place_entry(data_root, n_aug, 30, 150, 30, 50)

        control_class.place_text(data_root, "Specify augmentation operations", 20, 190, None, None)

        horizontal_flip = StringVar(data_root)
        hf_button = Checkbutton(data_root, text="Horizontal flip", variable=horizontal_flip, onvalue=True, offvalue=False)
        hf_button.place(bordermode=OUTSIDE, x=30, y=210, height=30, width=120)

        vertical_flip = StringVar(data_root)
        vf_button = Checkbutton(data_root, text="Vertical flip", variable=vertical_flip, onvalue=True, offvalue=False)
        vf_button.place(bordermode=OUTSIDE, x=150, y=210, height=30, width=120)

        control_class.place_text(data_root, "Width shift range", 30, 240, None, None)
        control_class.place_text(data_root, "(fraction of total width, if < 1, or pixels if >= 1)", 30, 260, None, None)
        control_class.place_entry(data_root, width_shift, 370, 250, 30, 50)

        control_class.place_text(data_root, "Height shift range", 30, 280, None, None)
        control_class.place_text(data_root, "(fraction of total height, if < 1, or pixels if >= 1)", 30, 300, None, None)
        control_class.place_entry(data_root, height_shift, 370, 290, 30, 50)

        control_class.place_text(data_root, "Shear range (Shear intensity)", 30, 340, None, None)
        control_class.place_entry(data_root, shear_range, 370, 330, 30, 50)

        control_class.place_text(data_root, "Rotation range (Degree range for random rotations)", 30, 380, None, None)
        control_class.place_entry(data_root, rotation_range, 370, 370, 30, 50)

        control_class.place_text(data_root, "Zoom range (Range for random zoom)", 30, 420, None, None)
        control_class.place_entry(data_root, zoom_range, 370, 410, 30, 50)

        check_weights = StringVar(data_root)
        Checkbutton(data_root, text="Create weight map", variable=check_weights, onvalue=True,
                    offvalue=False).place(bordermode=OUTSIDE, x=30, y=450, height=30, width=150)

        def generate_data():

            if dir_data_path.get() != "":

                if int(horizontal_flip.get()) == 1:
                    hf = True
                else:
                    hf = False

                if int(vertical_flip.get()) == 1:
                    vf = True
                else:
                    vf = False

                self.preprocess.splitImgs(dir_data_path.get(), tile_size.get(), tile_number.get())

                aug = Augment(dir_data_path.get(), shear_range.get(), rotation_range.get(), zoom_range.get(),
                              hf, vf, width_shift.get(), height_shift.get())

                if int(check_weights.get()) == 1:
                    wmap=True
                else:
                    wmap=False

                aug.start_augmentation(imgnum=n_aug.get(), wmap=wmap)
                aug.splitMerge(wmap=wmap)

                mydata = Create_npy_files(dir_data_path.get())

                mydata.create_train_data(wmap, tile_size.get(),  tile_size.get())

                tkinter.messagebox.showinfo("Done", "Augmented data successfully generated", parent=data_root)

            else:
                tkinter.messagebox.showinfo("Error", "Entries missing or not correct", parent=data_root)


        control_class.place_button(data_root, "Start data augmentation", generate_data, 150, 500, 30, 150)


    # Window: Train model
    def cont_training(self, old_window):

        """
        Train model

        :param old_window:
        :return:
        """

        old_window.destroy()

        cont_training = Tk()

        self.new_window(cont_training, "BioSegNet Navigator - Training", 500, 520)
        self.small_menu(cont_training)

        dir_data_path_train = StringVar(cont_training)
        epochs = IntVar(cont_training)
        balancer = DoubleVar(cont_training)
        learning_rate = DoubleVar(cont_training)
        batch_size = IntVar(cont_training)
        popup_var = StringVar(cont_training)
        popup_newex_var = StringVar(cont_training)
        model_name = StringVar(cont_training)
        use_weight_map = StringVar(cont_training)

        def askopendir_train():

            set_dir_data_path = tkinter.filedialog.askdirectory(parent=cont_training, title='Choose a directory')
            dir_data_path_train.set(set_dir_data_path)

            mydata = Create_npy_files(dir_data_path_train.get())

            try:

                zero_perc, fg_bg_ratio = mydata.check_class_balance()

                text = "Average percentage of background pixels in augmented label data: " + str(round(zero_perc*100,2))
                control_class.place_text(cont_training, text, 30, 360, None, None)

                text2 = "Foreground to background pixel ratio: 1 to " + str(fg_bg_ratio) + " "*30
                control_class.place_text(cont_training, text2, 30, 380, None, None)

                popup_newex_var.set("New")
                popupMenu_new_ex = OptionMenu(cont_training, popup_newex_var, *set(["New", "Existing"]))
                popupMenu_new_ex.place(bordermode=OUTSIDE, x=30, y=90, height=30, width=100)

                weight_images = os.listdir(dir_data_path_train.get() + os.sep + "aug_weights")

                if len(weight_images) == 0:

                    control_class.place_text(cont_training, "No weight map images detected.", 30, 280, 30, 180)

                    use_weight_map.set(0)

                else:

                    Checkbutton(cont_training, text="Use weight map", variable=use_weight_map, onvalue=True,
                                offvalue=False).place(bordermode=OUTSIDE, x=30, y=250, height=30, width=120)

                    text_bs = "When using a weight map for training, use a lower batch size"
                    control_class.place_text(cont_training, text_bs, 30, 275, None, None)
                    control_class.place_text(cont_training, "to not overload your GPU/CPU memory", 30, 290, None, None)

                    control_class.place_text(cont_training, "Class balance weight factor", 30, 325, None, None)
                    control_class.place_entry(cont_training, balancer, 250, 320, 30, 50)

            except:

                text_er = "Error: Please choose the BioSegNet Project directory"
                control_class.place_text(cont_training, text_er, 500, 30, 20, 380)

        text = "Select BioSegNet Project directory"
        control_class.place_browse(askopendir_train, text, dir_data_path_train, 20, 10, None, None, cont_training)

        control_class.place_text(cont_training, "Train new or existing model", 20, 70, None, None)


        def change_dropdown_newex(*args):

            if dir_data_path_train.get() != '':

                if popup_newex_var.get() == "New":

                    model_name.set("")

                    control_class.place_entry(cont_training, model_name, 333, 87, 33, 153)

                    text_mn = "Enter model name\n(without file extension)  "
                    control_class.place_text(cont_training, text_mn, 130, 90, 25, 200)

                else:

                    file_list = os.listdir(dir_data_path_train.get())
                    new_list = [i for i in file_list if ".hdf5" in i and not ".csv" in i]

                    if len(new_list) != 0:

                        control_class.place_text(cont_training, "Found the following model files  ", 140, 85, 35, 210)

                        model_name.set(new_list[0])
                        model_name_popupMenu = OptionMenu(cont_training, model_name, *set(new_list))
                        model_name_popupMenu.place(bordermode=OUTSIDE, x=335, y=87, height=35, width=150)

                        def change_dropdown(*args):
                            pass

                        model_name.trace('w', change_dropdown)

                    else:
                        control_class.place_text(cont_training, "No model found", 150, 90, 25, 150)

        popup_newex_var.trace('w', change_dropdown_newex)

        control_class.place_text(cont_training, "Number of epochs", 30, 140, None, None)
        control_class.place_entry(cont_training, epochs, 250, 135, 30, 50)

        control_class.place_text(cont_training, "Learning rate", 30, 180, None, None)
        control_class.place_entry(cont_training, learning_rate, 250, 175, 30, 50)

        control_class.place_text(cont_training, "Batch size", 30, 220, None, None)
        control_class.place_entry(cont_training, batch_size, 250, 215, 30, 50)

        popup_var.set("GPU")
        Label(cont_training, text="Train on", bd=1).place(bordermode=OUTSIDE, x=30, y=410)
        popupMenu_train = OptionMenu(cont_training, popup_var, *set(["GPU", "CPU"]))
        popupMenu_train.place(bordermode=OUTSIDE, x=30, y=430, height=30, width=100)


        def start_training():

            if dir_data_path_train.get() != "" and use_weight_map.get() != "" and epochs.get() != 0 and learning_rate.get() != 0 \
                    and batch_size.get() !=0 and model_name.get() != "":

                if int(use_weight_map.get()) == 1:
                    weight_map = True
                    bs = 1
                else:
                    weight_map = False
                    bs = batch_size.get()

                tile_size, y, x, tiles_list, images_list = self.get_image_info(dir_data_path_train.get(), False, False)

                train_biosegnet = BioSegNet(dir_data_path_train.get(), img_rows=tile_size, img_cols=tile_size,
                                              org_img_rows=y, org_img_cols=x)

                set_gpu_or_cpu = GPU_or_CPU(popup_var.get())
                set_gpu_or_cpu.ret_mode()

                #def train(self, epochs, wmap, vbal):
                train_biosegnet.train(epochs.get(), learning_rate.get(), bs, weight_map, balancer.get(),
                                       model_name.get(), popup_newex_var.get())

                tkinter.messagebox.showinfo("Done", "Training completed", parent=cont_training)


            else:
                tkinter.messagebox.showinfo("Error", "Entries missing or not correct", parent=cont_training)

        control_class.place_button(cont_training, "Start training", start_training, 200, 470, 30, 100)


    # Window: Model prediction
    def cont_prediction(self, old_window):

        """
        :param old_window:
        :return:
        """

        old_window.destroy()

        cont_prediction_window = Tk()

        self.new_window(cont_prediction_window, "BioSegNet Navigator - Prediction", 500, 350)
        self.small_menu(cont_prediction_window)

        dir_data_path_prediction = StringVar(cont_prediction_window)
        popup_var = StringVar(cont_prediction_window)
        popupvar_meas = StringVar(cont_prediction_window)
        batch_var = StringVar(cont_prediction_window)
        model_name = StringVar(cont_prediction_window)
        dir_data_path_test_prediction = StringVar(cont_prediction_window)
        found = IntVar()
        found.set(0)

        def askopendir_pred():

            set_dir_data_path = tkinter.filedialog.askdirectory(parent=cont_prediction_window, title='Choose a directory')
            dir_data_path_prediction.set(set_dir_data_path)

            if dir_data_path_prediction.get() != "":

                file_list = os.listdir(dir_data_path_prediction.get())
                new_list = [i for i in file_list if ".hdf5" in i and not ".csv" in i]

                if len(new_list) != 0:

                    found.set(1)

                    control_class.place_text(cont_prediction_window, "Found the following model files", 40, 60, 35, 190)

                    model_name.set(new_list[0])
                    model_name_popupMenu = OptionMenu(cont_prediction_window, model_name, *set(new_list))
                    model_name_popupMenu.place(bordermode=OUTSIDE, x=230, y=63, height=30, width=200)

                else:
                    control_class.place_text(cont_prediction_window, "No model found", 40, 60, 35, 360)

        text = "Select BioSegNet Project directory"
        control_class.place_browse(askopendir_pred, text, dir_data_path_prediction, 20, 10, None, None, cont_prediction_window)


        def askopendir_test_pred():

            set_dir_data_path_test = tkinter.filedialog.askdirectory(parent=cont_prediction_window,
                                                                     title='Choose a directory')
            dir_data_path_test_prediction.set(set_dir_data_path_test)

        text_s = "Select folder containing 8-bit images to be segmented" + " " * 30
        control_class.place_browse(askopendir_test_pred, text_s, dir_data_path_test_prediction, 20, 100, None, None,
                                   cont_prediction_window)

        def start_prediction():

            if dir_data_path_prediction.get() != "" and found.get() == 1 and dir_data_path_test_prediction.get() != "":

                tile_size = int(model_name.get().split("_")[-2])

                if batch_var.get() == "One folder":

                    y, x = self.get_image_info(dir_data_path_test_prediction.get(), True, False)

                else:

                    y, x = self.get_image_info(dir_data_path_test_prediction.get(), True, True)

                control_class.prediction(dir_data_path_test_prediction.get(), dir_data_path_prediction.get(), "",
                                         model_name.get(), batch_var.get(), popup_var.get(), popupvar_meas.get(),
                                         tile_size, y, x, cont_prediction_window)

            else:

                tkinter.messagebox.showinfo("Error", "Entries not completed", parent=cont_prediction_window)

        control_class.place_prediction_text(batch_var, popup_var, popupvar_meas, cont_prediction_window)

        control_class.place_button(cont_prediction_window, "Start prediction", start_prediction, 360, 290, 30, 110)

    ##########################################

    # Start new project window
    ##########################################

    def start_new_project(self):

        root.quit()

        start_root = Tk()

        self.new_window(start_root, "BioSegNet Navigator - Start new project", 500, 320)

        project_name = StringVar(start_root)
        dirpath = StringVar(start_root)
        orgpath = StringVar(start_root)
        labpath = StringVar(start_root)

        def askopendir():
            set_dirpath = tkinter.filedialog.askdirectory(parent=start_root, title='Choose a directory')
            dirpath.set(set_dirpath)

        def askopenorg():
            set_orgpath = tkinter.filedialog.askdirectory(parent=start_root, title='Choose a directory')
            orgpath.set(set_orgpath)

        def askopenlab():
            set_labpath = tkinter.filedialog.askdirectory(parent=start_root, title='Choose a directory')
            labpath.set(set_labpath)

        self.small_menu(start_root)

        control_class.place_text(start_root, "Select project name", 15, 10, None, None)
        control_class.place_entry(start_root, project_name, 25, 30, 30, 350)

        text = "Select directory in which BioSegNet project files should be generated"
        entry = control_class.place_browse(askopendir, text, dirpath, 15, 70, None, None, start_root)

        text = "Select directory in which 8-bit raw images are stored"
        entry_org = control_class.place_browse(askopenorg, text, orgpath, 15, 130, None, None, start_root)

        text = "Select directory in which ground truth (hand-labelled) images are stored"
        entry_lab = control_class.place_browse(askopenlab, text, labpath, 15, 190, None, None, start_root)


        def generate():

            str_dirpath = entry.get()
            str_orgpath = entry_org.get()
            str_labpath = entry_lab.get()

            control_class.generate_project_folder(False, project_name.get(), str_dirpath, str_orgpath, str_labpath,
                                                  start_root)

        control_class.place_button(start_root, "Generate", generate, 215, 260, 50, 70)

        start_root.mainloop()

    ##########################################

    # Continue working on existing project navigation window
    ##########################################

    def cont_project(self):

        cont_root = Tk()

        self.new_window(cont_root, "BioSegNet Navigator - Continue", 300, 200)
        self.small_menu(cont_root)

        h = 50
        w = 150

        control_class.place_button(cont_root, "Create augmented data", lambda: self.cont_data(cont_root), 87, 10, h, w)
        control_class.place_button(cont_root, "Train model", lambda: self.cont_training(cont_root), 87, 70, h, w)
        control_class.place_button(cont_root, "Model prediction", lambda: self.cont_prediction(cont_root), 87, 130, h, w)

    ##########################################


class EasyMode(Control):

    """
    Contains the following functionalities, in which most parameters can be adjusted by the user:

    Predict on pretrained model: select folder with raw 8-bit images and pretrained model to segment images
    Finetune pretrained model: if prediction on pretrained model fails, model can be trained (finetuned) on new data
    to improve segmentation accuracy
    """

    def __init__(self):
        Control.__init__(self)

    preprocess = Preprocess()


    # Window: Predict on pretrained model
    def predict_pretrained(self):

        """
        :return:
        """

        p_pt_root = Tk()

        self.new_window(p_pt_root, "BioSegNet Navigator - Predict using pretrained model", 500, 350)
        self.small_menu(p_pt_root)

        datapath = StringVar(p_pt_root)
        modelpath = StringVar(p_pt_root)
        popupvar = StringVar(p_pt_root)
        popupvar_meas = StringVar(p_pt_root)
        batch_var = StringVar(p_pt_root)

        def askopendata():
            set_datapath = tkinter.filedialog.askdirectory(parent=p_pt_root, title='Choose a directory')
            datapath.set(set_datapath)

        def askopenmodel():
            set_modelpath = tkinter.filedialog.askopenfilename(parent=p_pt_root, title='Choose a file')
            modelpath.set(set_modelpath)

        #### browse for raw image data
        text = "Select directory in which 8-bit raw images are stored"
        control_class.place_browse(askopendata, text, datapath, 15, 20, None, None, p_pt_root)
        ####

        #### browse for pretrained model
        text = "Select pretrained model file"
        control_class.place_browse(askopenmodel, text, modelpath, 15, 90, None, None, p_pt_root)
        ####

        control_class.place_prediction_text(batch_var, popupvar, popupvar_meas, p_pt_root)


        def start_prediction_pretrained():

            if datapath.get() != "" and modelpath.get() != "":

                # model file must have modelname_tilesize_
                tile_size = int(modelpath.get().split("_")[-2])

                model_path, model_file = os.path.split(modelpath.get())

                if batch_var.get() == "One folder":

                    y, x = self.get_image_info(datapath.get(), True, False)

                else:

                    y, x = self.get_image_info(datapath.get(), True, True)

                control_class.prediction(datapath.get(), datapath.get(), modelpath.get(), model_file, batch_var.get(),
                                         popupvar.get(), popupvar_meas.get(), tile_size, y, x, p_pt_root)

            else:

                tkinter.messagebox.showinfo("Error", "Entries not completed", parent=p_pt_root)


        control_class.place_button(p_pt_root, "Start prediction", start_prediction_pretrained, 360, 300, 30, 110)

    def pre_finetune_pretrained(self):

        pre_ft_pt_root = Tk()

        self.new_window(pre_ft_pt_root, "BioSegNet Navigator - Finetune pretrained model", 250, 380)
        self.small_menu(pre_ft_pt_root)

        control_class.place_button(pre_ft_pt_root, "New", easy_mode.new_finetune_pretrained, 45, 50, 130, 150)
        control_class.place_button(pre_ft_pt_root, "Existing", easy_mode.cont_finetune_pretrained, 45, 200, 130, 150)


    def cont_finetune_pretrained(self):

        ex_ft_pt_root = Tk()

        self.new_window(ex_ft_pt_root, "BioSegNet Navigator - Continue finetuning pretrained model", 500, 230)
        self.small_menu(ex_ft_pt_root)

        ft_datapath = StringVar(ex_ft_pt_root)
        epochs = IntVar(ex_ft_pt_root)
        popupvar = StringVar(ex_ft_pt_root)

        def askopenfinetune():
            set_ftdatapath = tkinter.filedialog.askdirectory(parent=ex_ft_pt_root, title='Choose the Finetune folder')
            ft_datapath.set(set_ftdatapath)

        #### browse for finetune folder
        text = "Select Finetune folder"
        control_class.place_browse(askopenfinetune, text, ft_datapath, 15, 20, None, None, ex_ft_pt_root)
        ####

        # set number of epochs
        control_class.place_text(ex_ft_pt_root, "Number of epochs", 20, 100, None, None)
        control_class.place_entry(ex_ft_pt_root, epochs, 250, 95, 30, 50)

        # set gpu or cpu training
        control_class.place_gpu(popupvar, 20, 130, ex_ft_pt_root)

        def start_training():

            if ft_datapath.get() != "":

                file_list = os.listdir(ft_datapath.get())
                model_list = [i for i in file_list if ".hdf5" in i and not ".csv" in i]

                # ignore tile size from project folder and rely instead on tile size noted on model file
                tile_size, y, x, tiles_list, images_list  = self.get_image_info(ft_datapath.get(), False, False)
                tile_size = int(model_list[0].split("_")[-2])

                train_biosegnet = BioSegNet(ft_datapath.get(), img_rows=tile_size,
                                              img_cols=tile_size, org_img_rows=y, org_img_cols=x)

                set_gpu_or_cpu = GPU_or_CPU(popupvar.get())
                set_gpu_or_cpu.ret_mode()

                mydata = Create_npy_files(ft_datapath.get())
                zero_perc, fg_bg_ratio = mydata.check_class_balance()

                learning_rate = 1e-4
                batch_size = 1
                balancer = 1/fg_bg_ratio


                if "weight_map" in model_list[0]:
                    wmap = True
                else:
                    wmap = False

                train_biosegnet.train(epochs.get(), learning_rate, batch_size, wmap, balancer, model_list[0], "Existing")

                modelname = model_list[0].split(".hdf5")[0]

                control_class.automatic_eval_train(ft_datapath.get() + os.sep + modelname + "training_log.csv")

                tkinter.messagebox.showinfo("Done", "Training / Finetuning completed", parent=ex_ft_pt_root)

            else:

                tkinter.messagebox.showinfo("Error", "Entries missing or not correct", parent=ex_ft_pt_root)

        control_class.place_button(ex_ft_pt_root, "Start training", start_training, 200, 190, 30, 100)


    # Window: Finetune pretrained model
    def new_finetune_pretrained(self):

        """
        :return:
        """

        ft_pt_root = Tk()

        self.new_window(ft_pt_root, "BioSegNet Navigator - Finetune pretrained model", 500, 450)
        self.small_menu(ft_pt_root)

        folder_name = StringVar(ft_pt_root)
        img_datapath = StringVar(ft_pt_root)
        label_datapath = StringVar(ft_pt_root)
        modelpath = StringVar(ft_pt_root)
        epochs = IntVar(ft_pt_root)
        popupvar = StringVar(ft_pt_root)

        def askopenimgdata():
            set_imgdatapath = tkinter.filedialog.askdirectory(parent=ft_pt_root, title='Choose a directory')
            img_datapath.set(set_imgdatapath)

        def askopenlabdata():
            set_labdatapath = tkinter.filedialog.askdirectory(parent=ft_pt_root, title='Choose a directory')
            label_datapath.set(set_labdatapath)

        def askopenmodel():
            set_modelpath = tkinter.filedialog.askopenfilename(parent=ft_pt_root, title='Choose a file')
            modelpath.set(set_modelpath)

        # set finetune folder name
        control_class.place_text(ft_pt_root, "Enter Finetune folder name", 15, 20, None, None)
        control_class.place_entry(ft_pt_root, folder_name, 25, 40, 30, 350)

        # browse for raw image data
        text = "Select directory in which 8-bit raw images are stored"
        control_class.place_browse(askopenimgdata, text, img_datapath, 15, 80, None, None, ft_pt_root)

        # browse for labelled data
        text = "Select directory in which ground truth (hand-labelled) images are stored"
        control_class.place_browse(askopenlabdata, text, label_datapath, 15, 150, None, None, ft_pt_root)

        # browse for pretrained model
        text = "Select pretrained model file"
        control_class.place_browse(askopenmodel, text, modelpath, 15, 220, None, None, ft_pt_root)

        # set number of epochs
        control_class.place_text(ft_pt_root, "Number of epochs", 30, 300, None, None)
        control_class.place_entry(ft_pt_root, epochs, 250, 295, 30, 50)


        # set gpu or cpu training
        control_class.place_gpu(popupvar, 20, 340, ft_pt_root)

        def start_training():

            l_temp = img_datapath.get().split(os.sep)
            parent_path = os.sep.join(l_temp[:-1])

            if folder_name.get() != "" and img_datapath.get() != "" and label_datapath.get() != "" \
                    and epochs.get() != 0 and modelpath.get() != "":

                # create project folder
                if not os.path.lexists(parent_path + os.sep + folder_name.get()):

                    control_class.generate_project_folder(True, folder_name.get(), parent_path, img_datapath.get(),
                                                          label_datapath.get(), ft_pt_root)

                else:

                    tkinter.messagebox.showinfo("Error", "Folder already exists", parent=ft_pt_root)


                # split label and 8-bit images into tiles
                tile_size = int(modelpath.get().split("_")[-2])

                y, x = self.get_image_info(img_datapath.get(), True, False)

                n_tiles = int(math.ceil(y/tile_size)*math.ceil(x/tile_size))

                self.preprocess.splitImgs(parent_path + os.sep + folder_name.get(), tile_size, n_tiles)

                # augment the tiles automatically (no user adjustments)

                shear_range = 0.1
                rotation_range = 180
                zoom_range = 0.1
                hf = True
                vf = True
                width_shift = 0.2
                height_shift = 0.2

                aug = Augment(parent_path + os.sep + folder_name.get(), shear_range, rotation_range, zoom_range, hf, vf,
                             width_shift, height_shift)

                # generate weight map if weight map model was chosen
                if "with_weight_map" in modelpath.get():
                    wmap = True
                else:
                    wmap = False

                n_aug = 50

                aug.start_augmentation(imgnum=n_aug, wmap=wmap, tile_size=tile_size)
                aug.splitMerge(wmap=wmap)

                # convert augmented files into array
                mydata = Create_npy_files(parent_path + os.sep + folder_name.get())

                mydata.create_train_data(wmap, tile_size, tile_size)

                train_biosegnet = BioSegNet(parent_path + os.sep + folder_name.get(), img_rows=tile_size,
                                              img_cols=tile_size, org_img_rows=y, org_img_cols=x)

                set_gpu_or_cpu = GPU_or_CPU(popupvar.get())
                set_gpu_or_cpu.ret_mode()

                zero_perc, fg_bg_ratio = mydata.check_class_balance()

                print("\nAverage percentage of background pixels: ", zero_perc)
                print("Foreground to background pixels ratio: ", fg_bg_ratio)
                print("Class balance factor: ", 1/fg_bg_ratio, "\n")

                learning_rate = 1e-4
                batch_size = 1
                balancer = 1/fg_bg_ratio

                # copy old model and rename to finetuned_model
                old_model_name = modelpath.get().split(os.sep)[-1]
                shutil.copy(modelpath.get(), parent_path + os.sep + folder_name.get() + os.sep + "finetuned_" + old_model_name)
                new_model_name = "finetuned_" + old_model_name

                train_biosegnet.train(epochs.get(), learning_rate, batch_size, wmap, balancer, new_model_name, "Finetuned_New")

                modelname = new_model_name.split(".hdf5")[0]
                control_class.automatic_eval_train(parent_path + os.sep + folder_name.get() + os.sep + modelname
                                                   + "training_log.csv")

                tkinter.messagebox.showinfo("Done", "Training / Finetuning completed", parent=ft_pt_root)



            else:

                tkinter.messagebox.showinfo("Error", "Entries missing or not correct", parent=ft_pt_root)

        control_class.place_button(ft_pt_root, "Start training", start_training, 200, 390, 30, 100)


if __name__ == '__main__':

    """
    Main (starting) window
    """

    control_class = Control()
    easy_mode = EasyMode()
    advanced_mode = AdvancedMode()

    root = Tk()

    control_class.new_window(root, "BioSegNet Navigator - Start", 400, 400)
    control_class.small_menu(root)

    # advanced mode

    control_class.place_text(root, "Advanced Mode", 248, 20, None, None)
    control_class.place_text(root, "Create your own model", 225, 40, None, None)

    control_class.place_button(root, "Start new project", advanced_mode.start_new_project, 215, 70, 130, 150)
    control_class.place_button(root, "Continue working on\nexisting project", advanced_mode.cont_project,
                               215, 220, 130, 150)

    # easy mode

    control_class.place_text(root, "Easy Mode", 90, 20, None, None)
    control_class.place_text(root, "Use a pretrained model", 55, 40, None, None)

    control_class.place_button(root, "Predict on\npretrained model", easy_mode.predict_pretrained, 45, 70, 130, 150)
    control_class.place_button(root, "Finetune\npretrained model", easy_mode.pre_finetune_pretrained, 45, 220, 130, 150)

    root.mainloop()



