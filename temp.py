
def prediction(self, datapath, modelpath, model_file, batch_var, popupvar, popupvar_meas, tile_size, y, x, n_tiles,
               window):

    pred_mitosegnet = MitoSegNet(datapath.get, img_rows=tile_size, img_cols=tile_size, org_img_rows=y,
                                 org_img_cols=x)

    set_gpu_or_cpu = GPU_or_CPU(popupvar)
    set_gpu_or_cpu.ret_mode()

    # n_tiles = int(math.ceil(y / tile_size) * math.ceil(x / tile_size))

    if popupvar_meas == "No measurements":
        datatype = False

    elif popupvar_meas == "CSV Table":
        datatype = "CSV"

    else:
        datatype = "Excel"


    if batch_var == "One folder":

        if not os.path.lexists(datapath + os.sep + "Prediction"):
            os.mkdir(datapath + os.sep + "Prediction")

        pred_mitosegnet.predict(datapath, False, tile_size, n_tiles, model_file, modelpath)

        control_class.get_measurements(datapath, datapath + os.sep + "Prediction", datatype,
                                       False)

    else:

        for subfolders in os.listdir(datapath):

            if not os.path.lexists(datapat h+ os.sep + subfolders + os.sep + "Prediction"):
                os.mkdir(datapath + os.sep + subfolders + os.sep + "Prediction")

            pred_mitosegnet.predict(datapath + os.sep + subfolders, False,
                                    tile_size, n_tiles, model_file, modelpath)

            labelpath = datapath + os.sep + subfolders + os.sep + "Prediction"

            control_class.get_measurements(datapat h+ os.sep + subfolders, labelpath, datatype,
                                           subfolders)

    tkinter.messagebox.showinfo("Done", "Prediction successful! Check " + datapath + os.sep +
                                "Prediction" + " for segmentation results", parent=window)