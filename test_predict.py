#!/usr/bin/env python
# coding: utf-8

import h5py
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import tensorflow as tf
import sys
import pandas as pd

def characterCrop(img, charBB):
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
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg+ dst
    dst2= np.where(dst2==(255,255,255), np.mean(croped, axis=(0,1), dtype=int).astype(np.uint8),dst2)
    
    return dst2

def readDataset(file_name, test=False):
    images = []
    db = h5py.File(file_name, 'r')
    im_names = list(db['data'].keys())
    cShape = (32, 32)
    wShape = (105, 105)
    letters = set('a')
    characterCount = 0
    for im in im_names:
        img = db['data'][im][:]
        if not test:
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
                    #theta_horizontal = 0
                    shear = 0
                    if (charBB[:,:,charIndx][1,0]>charBB[:,:,charIndx][1,3]):
                        shear = 180
                        theta_horizontal=180+theta_horizontal
                    aligned = tf.keras.preprocessing.image.apply_affine_transform(o, theta=-theta_horizontal, shear=shear)
                    resized_aligned = cv2.resize(aligned, cShape)
                    resized_aligned_gray = cv2.cvtColor(resized_aligned, cv2.COLOR_BGR2GRAY)
                    
                    if test:
                        characters.append({ 'char_txt': txt[i][j:j+1].decode('UTF-8'),'croped': o, 'resized_aligned': resized_aligned_gray})
                    else:
                        characters.append({ 'char_txt': txt[i][j:j+1].decode('UTF-8'),'croped': o, 'resized_aligned': resized_aligned_gray, 'font': font[charIndx].decode('UTF-8')})
                    characterCount +=1
                charIndx += 1
            wrd = txt[i].decode('UTF-8')
            letters = letters.union(set(wrd))
            wordImg = characterCrop(img, wordBB[:,:,i])
            if not 0 in wordImg.shape:
                wordImgRsz = cv2.resize(wordImg, wShape)
            words.append({'wordBB': wordBB[:,:,i], 'txt':txt[i].decode('UTF-8'), 'characters': characters, 'word': wordImg, 'resized': wordImgRsz, 'charsCount': charIndx })
            del characters
        if test:
            images.append({'img': img, 'txt': txt, 'words': words, 'im_name': im, 'wordBB': wordBB, 'charBB': charBB})
        else:
            images.append({'img': img, 'txt': txt, 'words': words, 'im_name': im, 'fonts': font, 'wordBB': wordBB, 'charBB': charBB})
    return images

def plotFonts(i):
    """
    plot the fonts per character in the image
    from the predictions
    
    Skylark - blue
    Sweet Puppy - green
    Ubuntu - red
    """
    img = images[i]['img']
    charBB = images[i]['charBB']
    wordBB = images[i]['wordBB']
    font_name = ['Skylark', 'Sweet Puppy', 'Ubuntu']
    fonts = predictions[predictions['image'] == images[i]['im_name']][['Skylark', 'Sweet Puppy', 'Ubuntu Mono']]
    fonts['index'] = list(range(0,len(fonts)))
    fonts = fonts.set_index('index')
    nC = charBB.shape[-1]
    plt.figure(figsize=(8,6))
    plt.imshow(img)
    for b_inx in range(nC):
        if fonts.loc[b_inx]['Skylark'] == 1:
            color = 'b'
        elif fonts.loc[b_inx]['Sweet Puppy'] == 1:
            color = 'g'
        else:
            color = 'r'
        bb = charBB[:,:,b_inx]
        x = np.append(bb[0,:], bb[0,0])
        y = np.append(bb[1,:], bb[1,0])
        plt.plot(x, y, color)
    # plot the word's BB:
    nW = wordBB.shape[-1]
    for b_inx in range(nW):
        bb = wordBB[:,:,b_inx]
        x = np.append(bb[0,:], bb[0,0])
        y = np.append(bb[1,:], bb[1,0])
        plt.plot(x, y, 'k')
    plt.title("blue - Skylark, green - Sweetpuppy, red - Ubuntu")

def predictTestset(images_file, model, pathCSV='char_font_preds_project'):
    """
    The procedure accepts dataset file in h5 format containing characters
    The function creates excel with font prediction per character
    using the following format
    ROW - IMAGE NAME - CHARACTER - FONT BINARY PREDICTION 1X3
    """
    images = readDataset(images_file, test=True)
    chars = []
    text = []
    imgs_name = []
    words = []
    fontClasses = ['Ubuntu Mono', 'Skylark', 'Sweet Puppy']
    for i in range(len(images)):
        for w in range(len(images[i]['words'])):
            for c in range(len(images[i]['words'][w]['characters'])):
                text.append(images[i]['words'][w]['characters'][c]['char_txt'])
                chars.append(images[i]['words'][w]['characters'][c]['resized_aligned'])
                words.append(w)
                imgs_name.append(images[i]['im_name'])
    chars = np.array(chars)
    text = np.array(text)
    shapeT = chars.shape
    chars_norm = chars.reshape(shapeT[0], shapeT[1], shapeT[2], 1)/255.0
    predictions = model.predict(chars_norm)
    
    df = pd.DataFrame(predictions)
    df = df.rename(columns={0: fontClasses[0], 1: fontClasses[1], 2: fontClasses[2]})
    df['image'] = imgs_name
    df['char'] = text
    df['word'] = words
    
    vote = df.groupby(['image', 'word']).sum().idxmax(axis=1)
    df['Skylark'] = 0
    df['Sweet Puppy'] = 0
    df['Ubuntu Mono'] = 0
    
    for i in range(0, len(df)):
        r = df.loc[i]
        df.loc[i, vote.loc[r['image'], r['word']]]=1
    
    df = df[['image', 'char', 'Skylark', 'Sweet Puppy', 'Ubuntu Mono']]
    df.to_csv(path_or_buf=pathCSV)
    return df    

path = 'test.h5'
outfile = 'char_font_pred.csv'

if sys.argv[1] == "-set":
    path = sys.argv[2]
if sys.argv[3] == "-outfile":
    outfile = sys.argv[4]
    
model = tf.keras.models.load_model("finalModel.h5")
predictions = predictTestset(path, model, pathCSV=outfile)

