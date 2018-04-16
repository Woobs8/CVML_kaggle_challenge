import numpy as np
import os, argparse, sys, re
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ImageReader import image_reader, create_image_lists
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
from scipy.misc import imresize, imsave
K.set_image_dim_ordering('th')


def augment_data(path_to_images, path_to_labels, save_image_path, number_images_per_class,lbls_path_and_file_name, target_size=None, resize_only=False):
    """ """
    # Get Image List
    img_list = create_image_lists(path_to_images)
    img_name_list = [os.path.split(x)[1][:-4] for x in img_list]

    # load image data
    X_train, y_train = image_reader(path_to_images,path_to_labels)

    # resize images to target size
    if not target_size is None:
        X_train = resize_images(X_train,(target_size[0],target_size[1]))
        
        # return now if only image resizing is required
        if resize_only:
            for idx, image in enumerate(X_train):
                im_path = os.path.join(save_image_path,'Image'+str(idx+1)+'.jpg')
                imsave(im_path, image)   
            return     

    # Data Augmentation To Use
    datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip = True,
        vertical_flip=True,
        fill_mode = "nearest",
        rotation_range=50,
        featurewise_center=True,
        data_format="channels_last")

    # convert from int to float
    X_train = X_train.astype('float32')
    
    # fit parameters from data
    datagen.fit(X_train)
    
    # Sample Count
    unique, counts = np.unique(y_train, return_counts=True)
    label_offset = int(unique[0])
    unique -= label_offset
    print("Starting Data Augmentation \n")
    for curr_class in unique:
        indices = np.where(y_train == curr_class+label_offset)[0].tolist()
        X_curr = X_train[indices,:,:,:]
        y_curr = y_train[indices]
        created_images=0
        for X_batch, y_batch in datagen.flow(X_curr, y_curr, 
                                            batch_size=10, 
                                            save_prefix="Image_C"+str(int(curr_class)),
                                            save_format="jpeg",
                                            save_to_dir=save_image_path):
            num_ = y_batch.shape
            created_images += num_[0]
            string_ = "Creating Images Of Class " + str(int(curr_class)) + ", " + str(created_images) + "/" + str(number_images_per_class) + "       "
            print(string_,end='\r',flush=True)
            if created_images >= number_images_per_class:
                break

    # Save Labels For Augmented Data
    img_list = create_image_lists(save_image_path)
    img_name_list = [os.path.split(x)[1][:-4] for x in img_list]
    new_lbls= create_labels(img_name_list)+label_offset
    np.save(lbls_path_and_file_name,new_lbls)
   

def create_labels(img_name_list):
    img_name_list = [os.path.split(x)[1][:-4] for x in img_name_list]
    labels = np.array([ int(img_lbl.split("_")[1][1:]) for img_lbl in img_name_list])
    return labels


def resize_images(image_data, target_size, interp='cubic'):
    num_samples, h, l, c = image_data.shape
    resized_data = np.zeros((num_samples, target_size[0], target_size[1], c))
    
    for i, image in enumerate(image_data):
        resized_data[i,:] = imresize(image,target_size,interp=interp)
        string_ = "Resizing images " + str(int(i+1)) + "/" + str(num_samples) + "       "
        print(string_,end='\r',flush=True)
    return resized_data

    
if __name__ == '__main__':
    import sys
    from Tools.DataReader import load_labels

    parser = argparse.ArgumentParser(prog='Augments Images and saves them on the local drive',
                                    description='''Program augments images and saves them on the local drive, in a specified folder.
                                                The specified number of images per class will approximatly be created
                                                Images are saved in the specified folder''')
    parser.add_argument('-path_to_images', 
                        help='Input path to images to be augmented',
                        default=None)
    
    parser.add_argument('-path_to_labels', 
                        help='Input path to images to be separated into classes',
                        default=None)
    
    parser.add_argument('-dir_path_to_save_images', 
                        help='Input path to folder where the images should be saved',
                        default=None)
    
    parser.add_argument('-number_images_per_class',
                        help='Total number of images that will be within each class (approximately)',
                        type=int,
                        default=0)

    parser.add_argument('-lbls_path_and_file_name',
                        help='File name of augmented data labels numpy file',
                        default="Lbls")

    parser.add_argument('-target_size',
                        help='Output image dimensions (h, w)',
                        nargs=2,
                        type=int,
                        required=False)

    parser.add_argument('-resize_only', 
                        help='Only perform image resizing - no other augmentation',
                        action="store_true")  

    #  Parse Arguments
    args = parser.parse_args()

    if (args.path_to_images or args.path_to_labels or args.dir_path_to_save_images or args.number_images_per_class) == None:
        raise ValueError("Didnt receive all arguments needed")
    # Path is a data file
    if os.path.exists(args.path_to_images):
        if not os.path.exists(args.dir_path_to_save_images):
            os.makedirs(args.dir_path_to_save_images )
        augment_data(args.path_to_images, args.path_to_labels, args.dir_path_to_save_images, args.number_images_per_class,args.lbls_path_and_file_name, target_size=args.target_size, resize_only=args.resize_only)
    else:
        print("Error: file '" + args.path_to_images + "' not found")
        exit(1)