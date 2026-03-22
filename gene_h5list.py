import os
import argparse
parser = argparse.ArgumentParser(description="Generate Text")
parser.add_argument("--h5_image", type=str, default="../train_640geo", help="name of the training data")
parser.add_argument("--h5_list", type=str, default="../train_640geo_dir.txt", help="name of the generated text")
args = parser.parse_args()
for files in os. listdir(args.h5_image):
    subfile_path = args.h5_image + '/' + files
    #print(subfile_path)
    for subfiles in os.listdir(subfile_path):
     #   subsubfile_path = subfile_path + '/' + subfiles
        img_path_gt = files + '/' + subfiles + '/' + 'gt.h5'
        with open(args.h5_list,"a") as f:
            f.write(str(img_path_gt))
            f.write('\n')
