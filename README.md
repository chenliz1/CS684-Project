# CS684-Project
## The repo for CS684 Final Project.

## Validation data:
1. Download http://www.cvlibs.net/downloads/depth_devkit.zip
2. Unzip depth_selection folder to "CS684-Project/data/"

## Training data:
1. Download https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip
2. Download https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0071/2011_09_29_drive_0071_sync.zip
3. Unzip. In "2011_09_**" folders, move "2011_09_**_drive_****_sync" folders to "CS684-Project/data/"

## Dataser Preparation:
1. The files in "CS684-Project/data/" look like this:<br>
<p>
 |--- /CS684-Project<br>
 |------ /data <br>
 |--------- /2011_**_0001_sync<br>
 |--------- /2011_**_0071_sync<br>
 |--------- /depth_selection<br>
 |--------- datasetCreator.py<br>
 |--------- read_depth.py<br></p>

2. Under "CS684-Project/data/", run datasetCreator.py.
