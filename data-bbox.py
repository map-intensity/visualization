import numpy as np
import itertools
import os
import cv2

def rename():
    #this function will make "IMG_X.jpg" to IMG_XXX.jpg
    directory1 = "data/part_B_final/test_data/images" # replace "." with the path to the directory containing the files
    directory2 = "data/part_B_final/train_data/images" # replace "." with the path to the directory containing the files

    for directory in [directory1,directory2]:
        for filename in os.listdir(directory):
            if filename.endswith(".jpg"):
                old_path = os.path.join(directory, filename)
                base_name, ext = os.path.splitext(filename)
                # extract number from "IMG_1234.jpg"
                # and pad it with zeros to 4 digits
                number = base_name[4:]
                newnumber = number.zfill(3)
                new_name = "IMG_"+newnumber + ext
                new_path = os.path.join(directory, new_name)
                os.rename(old_path, new_path)
                print("Renamed {} to {}".format(old_path, new_path))
    print("Done")

#Step1: output the object detection using yolo-8 to create the raw gt (.txt) for the test and train data
#run the following code to use yolo-v8
"""
cd ./proof-concept-work-trainbbox
python yolov8_tracking/track.py --yolo-weights yolov8n.pt --source ./data/part_B_final/test_data/images/ --classes 0 --save-txt
python yolov8_tracking/track.py --yolo-weights yolov8n.pt --source ./data/part_B_final/train_data/images/ --classes 0 --save-txt
"""
# Then copy the txt files to data dir and process data for both train and test
# post-process the data
def getGTfile():
    """
    GT File formate(per line)
    frame_idx,bbox_left,bbox_top,bbox_w,bbox_h,bbox_area
    """
    test_targetFile = "./data/part_B_final/test_data/test.txt"
    test_OUTPUT_DIR_PATH = './data/part_B_final/test_data/gt/'

    f = open(test_targetFile,"r")
    objs = []
    for line in f:
        obj_dic = {}
        line = line.split(' ')
        obj_dic['frame_idx'] = int(line[0])
        obj_dic['bbox_left'] = int(line[2])
        obj_dic['bbox_top'] = int(line[3])
        obj_dic['bbox_w'] = int(line[4])
        obj_dic['bbox_h'] = int(line[5])
        obj_dic['bbox_weight'] = 1/(int(line[4]) * int(line[5]))
        objs.append(obj_dic)
    grouped_dicts = itertools.groupby(objs, lambda x: x["frame_idx"])
    clustered_dicts = [[d for d in g] for k, g in grouped_dicts if k]
    for frame_objs in clustered_dicts:
        output = []
        for obj in frame_objs:
            output.append([obj["bbox_left"],obj["bbox_top"],obj["bbox_w"],obj["bbox_h"],obj["bbox_weight"]])
        number = str(frame_objs[0]['frame_idx']).zfill(3)
        fname = test_OUTPUT_DIR_PATH + "IMG_"+ number +'.csv'        
        np.savetxt(fname,output, delimiter=',')
    f.close()

    train_targrtFile = "./data/part_B_final/train_data/train.txt"
    train_OUTPUT_DIR_PATH = './data/part_B_final/train_data/gt/'
    f = open(train_targrtFile,"r")
    objs = []
    for line in f:
        obj_dic = {}
        line = line.split(' ')
        obj_dic['frame_idx'] = int(line[0])
        obj_dic['bbox_left'] = int(line[2])
        obj_dic['bbox_top'] = int(line[3])
        obj_dic['bbox_w'] = int(line[4])
        obj_dic['bbox_h'] = int(line[5])
        obj_dic['bbox_weight'] = 1/(int(line[4]) * int(line[5]))
        objs.append(obj_dic)
    grouped_dicts = itertools.groupby(objs, lambda x: x["frame_idx"])
    clustered_dicts = [[d for d in g] for k, g in grouped_dicts if k]
    for frame_objs in clustered_dicts:
        output = []
        for obj in frame_objs:
            output.append([obj["bbox_left"],obj["bbox_top"],obj["bbox_w"],obj["bbox_h"],obj["bbox_weight"]])
        number = str(frame_objs[0]['frame_idx']).zfill(3)
        fname = train_OUTPUT_DIR_PATH + "IMG_"+ number +'.csv'
        np.savetxt(fname,output, delimiter=',')
    f.close()

def showbbox():
    # this function can get the bbox annotaion on the origan pic

    # Define the directory path containing the images and MOT format annotation files
    test_img_dir_path = "./data/part_B_final/test_data/images"
    test_gt_dir_path = "./data/part_B_final/test_data/gt"
    train_img_dir_path = "./data/part_B_final/train_data/images"
    train_gt_dir_path = "./data/part_B_final/train_data/gt"
    dirpath = [(test_img_dir_path,test_gt_dir_path),(train_img_dir_path,train_gt_dir_path)]
    for img_dir_path,gt_dir_path in dirpath:
        # Loop through each file in the directory
        for file_name in os.listdir(img_dir_path):
            # Check if the file is an image file
            if file_name.endswith(".jpg"):
                # Load the image
                img_path = os.path.join(img_dir_path, file_name)
                img = cv2.imread(img_path)
                
                # Load the corresponding MOT format annotation file
                txt_path = os.path.join(gt_dir_path, os.path.splitext(file_name)[0] + ".csv")
                if not os.path.exists(txt_path):
                    print(f"Annotation file not found for {file_name}")
                    continue
                with open(txt_path, "r") as f:
                    annotations = f.readlines()
                
                # Loop through each annotation in the file and draw a bounding box on the image
                for ann in annotations:
                    ann_parts = ann.strip().split(",")
                    if len(ann_parts) != 5:
                        print(f"Invalid annotation format in {txt_path}: {ann}")
                        continue
                    x, y, w, h, a = ann_parts
                    x, y, w, h = int(float(x)), int(float(y)), int(float(w)), int(float(h))
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Save the annotated image to a new file
                annotated_img_path = os.path.join(gt_dir_path, os.path.splitext(file_name)[0] + "_bbox_annotated.jpg")
                cv2.imwrite(annotated_img_path, img)

# getGTfile()
showbbox()
