import h5py
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw, ImageFont
import random
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import itertools

def characterCrop(img, charBB):
    """
    Crop character from image
    Use given bounding box to crop the character from image
    Cut rectangle from the image, the using mask over the bouding box
    black the background and repaint in average color of the character.

    Params
    @img - img to extract character from 
    """
    for i in range(4):
        if charBB[0, i] < 0:
            charBB[0, i] = 0
        if charBB[1, i] < 0:
            charBB[1, i] = 0
        if charBB[0, i] > img.shape[1]:
            charBB[0, i] = img.shape[1]
        if charBB[1, i] > img.shape[0]:
            charBB[1, i] = img.shape[0]
            
    pts = np.array(list((zip(charBB[0], charBB[1]))),dtype='int')
    
    
    #pts = np.array([[9,502],[24,501],[24,520],[9,521]])
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()
    
    ## (2) make mask
    pts = pts - pts.min(axis=0)
    
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    
    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    
    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg+ dst
    dst2= np.where(dst2==(255,255,255), np.mean(croped, axis=(0,1), dtype=int).astype(np.uint8),dst2)
    
    return dst2

def readDataset(file_name):
    """
    Read dataset from given h5 file.
    Create python list containing dictionary in the following form

    images_array [
        image -> {
            'img' - image original photo
            'txt' - array of text appears in the photo
            'im_name' - name of the image
            'fonts' - fonts in the image 
            'wordBB' - bounding boxes of words in the image
            'charBB' - bounding boxes of characters in the image
            'words' - [
                {
                    'wordBB' - bounding box of the word
                    'txt' - word text
                    'word' - croped image of the word
                    'characters' - [
                        {
                            'char_txt' - the character text 
                            'croped' - the character simply croped before any further data manipulation
                            'resized_aligned' - aligned character in size of 32x32 pixels
                            'font' - character's font
                        }
                    ]
                }
            ]
        }
    ]
    """
    images = []
    db = h5py.File(file_name, 'r')
    im_names = list(db['data'].keys())
    cShape = (32, 32)

    for im in im_names:
        img = db['data'][im][:]
        font = db['data'][im].attrs['font']
        txt = db['data'][im].attrs['txt']
        charBB = db['data'][im].attrs['charBB']
        wordBB = db['data'][im].attrs['wordBB']
        charIndx = 0
        
        words = []
        for i in range(0, len(txt)):
            characters = []
            for j in range(0, len(txt[i])):
                o = characterCrop(img, charBB[:,:,charIndx])
                if not 0 in o.shape:
                    theta_horizontal = math.degrees(math.atan2(charBB[:,:,charIndx][1,2]-charBB[:,:,charIndx][1,3], charBB[:,:,charIndx][0,2]-charBB[:,:,charIndx][0,3]))
                    shear = 0
                    if (charBB[:,:,charIndx][1,0]>charBB[:,:,charIndx][1,3]):
                        shear = 180
                        theta_horizontal=180+theta_horizontal
                    aligned = tf.keras.preprocessing.image.apply_affine_transform(o, theta=-theta_horizontal, shear=shear)
                    resized_aligned = cv2.resize(aligned, cShape)
                    resized_aligned_gray = cv2.cvtColor(resized_aligned, cv2.COLOR_BGR2GRAY)

                    characters.append({ 'char_txt': txt[i][j:j+1].decode('UTF-8'),'croped': o, 'resized_aligned': resized_aligned_gray, 'font': font[charIndx].decode('UTF-8')})
                charIndx += 1
            wordImg = characterCrop(img, wordBB[:,:,i])
            words.append({'wordBB': wordBB[:,:,i], 'txt':txt[i].decode('UTF-8'), 'characters': characters, 'word': wordImg })
            del characters
        images.append({'img': img, 'txt': txt, 'words': words, 'im_name': im, 'fonts': font, 'wordBB': wordBB, 'charBB': charBB})
    return images

images = readDataset('SynthText.h5')
images_val = readDataset('SynthText_val.h5')
images_added = readDataset('train.h5')
# Join all data together
images.extend(images_val)
images.extend(images_added)

# All known characters in the dataset
letters = ['!','"','#','$','%',"'",'(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[',']','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','|', '&', '~']
chars = []
y = []

# Create train and numeric target set by classes
fontClasses = ['Ubuntu Mono', 'Skylark', 'Sweet Puppy']
for i in range(len(images)):
    for w in range(len(images[i]['words'])):
        for c in range(len(images[i]['words'][w]['characters'])):
            chars.append(images[i]['words'][w]['characters'][c]['resized_aligned'])
            y.append(fontClasses.index(images[i]['words'][w]['characters'][c]['font']))
            
chars = np.array(chars)
y = np.array(y)

rng = list(range(len(chars)))
np.random.shuffle(rng)

train_x = chars[rng]
train_y = tf.keras.utils.to_categorical(y)[rng]
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32,1)),
    tf.keras.layers.LayerNormalization(axis=1 , center=True , scale=True),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(len(letters), activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss = 'mean_squared_error', optimizer='adam', metrics=['accuracy'])

shapex = train_x.shape
train_x_norm = train_x.reshape(shapex[0], shapex[1], shapex[2], 1)/255.0

# train model
history = model.fit(train_x_norm, train_y,
          epochs=20, batch_size=256)

model.save("CNN_font_original+val_alltrain.h5")