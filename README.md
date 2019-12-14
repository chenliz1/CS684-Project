# CS684-Project
## The repo for CS684 Final Project.

## Dataset Preparation:
1. Download http://www.cvlibs.net/download.php?file=data_depth_selection.zip
2. Unzip "depth_selection" folder to "CS684-Project/data/"
3. In "CS684-Project/data/", run:
```shell
wget -i kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```
4. The files in "CS684-Project/data/" look like this:<br>
<p>
 | /CS684-Project<br>
 |---- /data <br>
 |-------- datasetCreator.py<br>
 |-------- /depth_selection<br>
 |-------- /kitti_data<br>
 |-------- ......
</p>


7. In "CS684-Project/data/", run 
```shell 
python datasetCreator.py
```
baseline = 0.54

f = 7.215377e+02
