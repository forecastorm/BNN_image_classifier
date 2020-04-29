'''
import images from train_validation, stores in N X 1 X 28 X 28 format, extrat labels 

'''
import cv2
from sklearn.model_selection import train_test_split

# read in gray scale 
def load_image(file_path):
    return cv2.imread(file_path)


def extract_label(file_name):
    return 1 if "open" in file_name else 0 # open eyes are 1 & closed eyes are 0

#split image files, extract lables later 
def train_valid_test_split(image_files):
    train_files, test_files = train_test_split(image_files,test_size=0.2)
    train_files, valid_files = train_test_split(train_files,test_size=0.2)
    return train_files,valid_files,test_files

def preprocess_image(img): 
    img = cv2.resize(img,(28,28))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # binarize the image
    # (pixel < 127) = 0 
    # (plexel > 127) = 1
    #ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    return img/255