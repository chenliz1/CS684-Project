# CS684-Project
## The repo for CS684 Final Project.

## Validation data:
1. Download http://www.cvlibs.net/downloads/depth_devkit.zip
2. Unzip "depth_selection" folder to "CS684-Project/data/"

## Training data:
1. Download https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip
2. Download https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0071/2011_09_29_drive_0071_sync.zip
3. Unzip. In "2011_09_26" folders, move "2011_09_26_drive_0001_sync" folders to "CS684-Project/data/"
3. Unzip. In "2011_09_28" folders, move "2011_09_28_drive_0071_sync" folders to "CS684-Project/data/"
4. Download annotations: http://www.cvlibs.net/download.php?file=data_depth_annotated.zip
5. Unzip. Open the "2011_09_26_drive_0001_sync" folder in ""data_depth_annotated/train/".
6. Move "proj_depth" to the correspond folder "CS684-Project/data/2011_09_26_drive_0001_sync"
5. Open the "2011_09_28_drive_0071_sync" folder in ""data_depth_annotated/train/".
6. Move "proj_depth" to the correspond folder "CS684-Project/data/2011_09_28_drive_0071_sync"


## Dataser Preparation:
1. The files in "CS684-Project/data/" look like this:<br>
<p>
 | /CS684-Project<br>
 |---- /data <br>
 |-------- /depth_selection<br>
 |-------- datasetCreator.py<br>
 |-------- read_depth.py<br></p>
 |-------- /2011_09_26_drive_0001_sync<br>
 |------------ ...<br>
 |------------ /image_02<br>
 |------------ /proj_depth<br>
 |------------ ...<br>
 |-------- /2011_09_29_drive_0071_sync<br>
 |------------ ...<br>
 |------------ /image_02<br>
 |------------ /proj_depth<br>
 |------------ ...<br>


2. Under "CS684-Project/data/", run datasetCreator.py.
