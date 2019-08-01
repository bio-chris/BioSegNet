import os
from shutil import copyfile
import tkinter.messagebox

class CreateProject:

    def __init__(self, project_name):

        self.project_name = project_name

    def create_folders(self, path):

        if not os.path.lexists(path + os.sep + self.project_name):

            os.mkdir(path + os.sep + self.project_name)

            os.mkdir(path + os.sep + self.project_name + os.sep + "aug_label")
            os.mkdir(path + os.sep + self.project_name + os.sep + "aug_merge")
            os.mkdir(path + os.sep + self.project_name + os.sep + "aug_train")
            os.mkdir(path + os.sep + self.project_name + os.sep + "aug_weights")
            os.mkdir(path + os.sep + self.project_name + os.sep + "merge")
            os.mkdir(path + os.sep + self.project_name + os.sep + "npydata")

            os.mkdir(path + os.sep + self.project_name + os.sep + "train")

            os.mkdir(path + os.sep + self.project_name + os.sep + "train" + os.sep + "image")
            os.mkdir(path + os.sep + self.project_name + os.sep + "train" + os.sep + "label")

            os.mkdir(path + os.sep + self.project_name + os.sep + "train" + os.sep + "RawImgs")

            os.mkdir(path + os.sep + self.project_name + os.sep + "train" + os.sep + "RawImgs" + os.sep + "image")
            os.mkdir(path + os.sep + self.project_name + os.sep + "train" + os.sep + "RawImgs" + os.sep + "label")

            return True

        else:

            tkinter.messagebox.showinfo("Note", "MitoSegNet_Project folder already exists!")
            return False

    def copy_data(self, path, orgpath, labpath):

        image_dest_path = path + os.sep + self.project_name + os.sep + "train" + os.sep + "RawImgs" + os.sep + "image"
        label_dest_path = path + os.sep + self.project_name + os.sep + "train" + os.sep + "RawImgs" + os.sep + "label"

        file_list = os.listdir(orgpath)

        for files in file_list:

            copyfile(orgpath + os.sep + files, image_dest_path + os.sep + files)
            copyfile(labpath + os.sep + files, label_dest_path + os.sep + files)



