import numpy as np
import os
import argparse
import tensorflow as tf
import re
from shutil import move

def separate_classes_into_folders(labels):
    """ """
    imglist = create_image_lists(args.path_to_pictures)
    unique_lbls = np.unique(labels)
    # Create The Folders Corrosponding To Labels
    for lbl in unique_lbls:
        path = args.path_to_save_images+"/"+str(int(lbl))
        print("Creating Label folder: ")
        print(path)
        
        if not os.path.exists(path):
            os.makedirs(path)
    
        #Copy Images Into Class Separated Folders
        
        for idx in np.transpose(np.where(labels==lbl)):
            head, tail = os.path.split(imglist[int(idx)])
            move(imglist[int(idx)], path+"/"+tail)


def create_image_lists(image_dir):
    """ Builds a list of images from the file system.
    """
    # The root directory comes first, so skip it.
    is_root_dir = True
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(image_dir)
    tf.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
        file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
        file_list.extend(tf.gfile.Glob(file_glob))
    if not file_list:
        tf.logging.warning('No files found')
        return
    
    arg_sort = np.argsort([int(re.search("(\d*\.?\d)",val).group(0)) for val in file_list])
    imglist = [file_list[idx] for idx in arg_sort]
    return imglist

if __name__ == '__main__':
    import sys
    from Tools.DataReader import load_labels

    parser = argparse.ArgumentParser(prog='Separate images into folders sorted by classes',
                                    description='''Program separates images into folders sorted by the classes of the image.
                                                The folders are created at the location specified or if unspecified at the script location''')
    parser.add_argument('path_to_pictures', 
                        help='Input path to images to be separated into classes')
    
    parser.add_argument('path_to_labels', 
                        help='Input path and name of file containing the labels of the images')
    
    parser.add_argument('-path_to_save_images', 
                        help='Input path to folder where the images should be saved',
                        default=os.path.dirname(os.path.abspath(__file__)))

    #  Parse Arguments
    args = parser.parse_args()

    # Path is a data file
    if os.path.exists(args.path_to_pictures):
        if os.path.exists(args.path_to_labels):
            # Read labels from file
            labels = load_labels(args.path_to_labels)
        
        else:
            print("Error: file '" + args.path_to_labels + "' not found")
            exit(1)

    else:
        print("Error: file '" + args.path_to_pictures + "' not found")
        exit(1)
    
    separate_classes_into_folders(labels)
    