/*
 * Converts tif or non-tif images or stacks into max intensity projected tif files for subsequent MitoSegNet
 * segmentation
 * 
 */

in = getDirectory("Select directory in which images are stored"); 
names_org = getFileList(in);
save_path = in + File.separator + "MaxInt_Tiff_Files";

if (File.isDirectory(save_path)){
	print("Directory already exists");
}
else{
	File.makeDirectory(in + File.separator + "MaxInt_Tiff_Files");
}

setBatchMode(true); 
run("Bio-Formats Macro Extensions");

for (i = 0; i<names_org.length; i++){

	Ext.openImagePlus(in + File.separator + names_org[i]);
	getDimensions(width, height, channels, slices, frames);

	if(slices>1){
		run("Z Project...", "projection=[Max Intensity]");
		selectWindow("MAX_" + names_org[i]);		
	}	
	
	run("8-bit");

	saveAs("Tiff", save_path + File.separator + names_org[i]);
	run("Close All");

	run("Collect Garbage");
}
