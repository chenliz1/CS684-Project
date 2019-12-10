import glob
import os
from PIL import Image
import numpy as np

train_data_folders = ["2011_09_26_drive_0001_sync", "2011_09_29_drive_0071_sync"]
annotation = "data_depth_annotated/train/"
kitti = "kitti_data/2011_09_26"

val_img_path = "depth_selection/val_selection_cropped/image/"
val_gts_path = "depth_selection/val_selection_cropped/groundtruth_depth/"
target_val_path = "dataset/val/"
target_train_path = "dataset/train/"

def makeDirs():
    if not (os.path.isdir("dataset")):
        os.mkdir("dataset")

    if not (os.path.isdir("dataset/val")):
        os.mkdir("dataset/val")

    if not (os.path.isdir("dataset/train")):
        os.mkdir("dataset/train")

    for str in ["dataset/train", "dataset/val"]:
        for target in ["image_left", "image_right"]:
            if not (os.path.isdir(os.path.join(str, target))):
                os.mkdir(os.path.join(str, target))



def valImages(img_path, gts_path, target_path):
    imgL_list = list(glob.glob1(img_path, "*_02.png"))
    imgL_list.sort()

    imgR_list = list(glob.glob1(img_path, "*_03.png"))
    imgR_list.sort()

    # gtsL_list = list(glob.glob1(gts_path, "*_02.png"))
    # gtsL_list.sort()
    #
    # gtsR_list = list(glob.glob1(gts_path, "*_03.png"))
    # gtsR_list.sort()

    target_imgL_path = os.path.join(target_path, "image_left/")
    target_imgR_path = os.path.join(target_path, "image_right/")
    # target_gtsL_path = os.path.join(target_path, "groundtruth_left/")
    # target_gtsR_path = os.path.join(target_path, "groundtruth_right/")

    for i in range(len(imgL_list)):
        target_name = str(i).zfill(5)

        imgL = Image.open(os.path.join(img_path, imgL_list[i]))
        imgL = imgL.convert('RGB')
        imgL.save(os.path.join(target_imgL_path, target_name + ".jpg"))

        imgR = Image.open(os.path.join(img_path, imgR_list[i]))
        imgR = imgR.convert('RGB')
        imgR.save(os.path.join(target_imgR_path, target_name + ".jpg"))

        # gts = Image.open(os.path.join(gts_path, gts_list[i]))
        # gts.save(os.path.join(target_gts_path, target_name + ".png"))


def trainImages(kitti, annotation, target_path):
    train_data_folders = list(glob.glob1(kitti, "2011_09_26_drive_*_sync"))
    train_data_folders.sort()

    counter = 0
    for folder in train_data_folders:
        imgL_path = os.path.join(kitti, folder, "image_02/data")
        imgR_path = os.path.join(kitti, folder, "image_03/data")
        # gts_path = os.path.join(annotation, folder, "proj_depth/groundtruth/image_02/")

        imgL_list = list(glob.glob1(imgL_path, "*.png"))
        imgL_list.sort()
        imgR_list = list(glob.glob1(imgR_path, "*.png"))
        imgR_list.sort()
        # gts_list = list(glob.glob1(gts_path, "*.png"))
        # gts_list.sort()

        target_imgL_path = os.path.join(target_path, "image_left/")
        target_imgR_path = os.path.join(target_path, "image_right/")
        # target_gts_path = os.path.join(target_path, "groundtruth/")

        for i in range(len(imgL_list)):
            target_name = str(counter).zfill(5)

            imgL = Image.open(os.path.join(imgL_path, imgL_list[i]))
            imgL = imgL.convert('RGB')
            imgL.save(os.path.join(target_imgL_path, target_name + ".jpg"))

            imgR = Image.open(os.path.join(imgR_path, imgR_list[i]))
            imgR = imgR.convert('RGB')
            imgR.save(os.path.join(target_imgR_path, target_name + ".jpg"))

            # gts = Image.open(os.path.join(gts_path, gts_list[i]))
            # gts.save(os.path.join(target_gts_path, target_name + ".png"))
            counter += 1


makeDirs()
trainImages(kitti, annotation, target_train_path)
valImages(val_img_path, val_gts_path, target_val_path)
