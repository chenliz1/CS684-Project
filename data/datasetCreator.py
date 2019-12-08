import glob
import os
from PIL import Image

train_data_folders = ["2011_09_26_drive_0001_sync", "2011_09_29_drive_0071_sync"]

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
        for target in ["image", "groundtruth"]:
            if not (os.path.isdir(os.path.join(str, target))):
                os.mkdir(os.path.join(str, target))



def valImages(img_path, gts_path, target_path):
    img_list = list(glob.glob1(img_path, "*.png"))
    img_list.sort()
    gts_list = list(glob.glob1(gts_path, "*.png"))
    gts_list.sort()

    target_img_path = os.path.join(target_path, "image/")
    target_gts_path = os.path.join(target_path, "groundtruth/")

    for i in range(len(img_list)):
        target_name = str(i).zfill(5) + ".jpg"

        img = Image.open(os.path.join(img_path, img_list[i]))
        img = img.convert('RGB')
        img.save(os.path.join(target_img_path, target_name))

        gts = Image.open(os.path.join(gts_path, gts_list[i]))
        gts = gts.convert('L')
        gts.save(os.path.join(target_gts_path, target_name))


def trainImages(train_data_folders, target_path):
    counter = 0
    for folder in train_data_folders:
        img_path = os.path.join(folder, "image_02/")
        gts_path = os.path.join(folder, "proj_depth/groundtruth/image_02/")
        img_list = list(glob.glob1(img_path, "*.png"))
        img_list.sort()
        gts_list = list(glob.glob1(gts_path, "*.png"))
        gts_list.sort()

        target_img_path = os.path.join(target_path, "image/")
        target_gts_path = os.path.join(target_path, "groundtruth/")

        for i in range(len(img_list)):
            target_name = str(counter).zfill(5) + ".jpg"

            img = Image.open(os.path.join(img_path, img_list[i]))
            img = img.convert('RGB')
            img.save(os.path.join(target_img_path, target_name))

            gts = Image.open(os.path.join(gts_path, gts_list[i]))
            gts = gts.convert('L')
            gts.save(os.path.join(target_gts_path, target_name))
            counter += 1


makeDirs()
trainImages(val_img_path, val_gts_path, target_val_path)

