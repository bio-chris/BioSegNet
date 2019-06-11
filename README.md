# BioSegNet Documentation

---

## Before running the BioSegNet

### GPU usage

## Running the BioSegNet

### Easy Mode

If you are unfamiliar with deep learning concepts such as batch size, learning rate or augmentation operations, then it is recommended to use the Easy Mode, in which most of the deep learning parameters are pre-set. 

Predict on pretrained model

* Select directory in which 8-bit raw images are stored:
* Images have to be 8-bit, tif format and single plane. Use the macro MitoSegNet_PreProcessing.ijm for automated conversion of a large number of images (Prerequisite for macro usage is installation of Bio-Formats plugin on Fiji)
* Select pretrained_model_656.hdf5 (which can be found in installation folder)
* Depending if you have all images in one folder, or multiple set of images in sub-folders you can select to apply the model to one folder or multiple folders (Folder > Subfolder > Images)
* Select to predict on GPU or CPU 

Once all entries are filled, click on Start prediction and wait until a Done window opens to notify of the successful completion.

If segmentation with a pretrained model did not generate good results you can try to finetune the pretrained model to your own data

Finetune pretrained model

* Select directory in which 8-bit raw images are stored:
* Select directory in which hand-labelled (ground truth) images are stored 
* Select pretrained_model_656.hdf5
* Specify the number of epochs (defines the repetitions the training process will run through to learn how to segment the images based on the new input)

Once all entries are filled, click on Start training and wait until a Done window opens to notify of the successful completion. 
In the parent directory of the raw image and label image folder a Finetune_folder will be generated in which all the newly augmented data, image arrays and finetuned models will be stored.


### Advanced Mode

Start new project

Continue working on existing project

* Create augmented data
* Train model
* Model prediction

