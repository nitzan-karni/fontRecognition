#!/usr/bin/env python
# coding: utf-8

# In[2]:


import h5py
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import normalize, to_categorical
from keras.utils import to_categorical
from keras import callbacks
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , UpSampling2D ,Conv2DTranspose
from keras import backend as K
from keras import optimizers
import random
# In[4]:
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


# In[5]:
def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy


# In[6]:
def random_image_augmentation(images, thetaRange=90, shearList=[0, 0, 0, 0, 0, 0, 0, 0, 0, 180], noiseType='s&p'):
    # Data augmentation
    for i, img in enumerate(images):
        im = tf.keras.preprocessing.image.random_brightness(img, (0.7,1.3)).astype(np.uint8)
        im = tf.keras.preprocessing.image.random_zoom(im, (1, 1)).astype(np.uint8)
        theta = np.random.randint(-thetaRange, thetaRange)
        #theta = 0
        shear = random.sample(shearList, 1)[0]
        #shear = 0
        im = tf.keras.preprocessing.image.apply_affine_transform(im, theta=theta, shear=shear)
        im = noisy(noiseType,im)
        images[i] = im
    return images
        


# In[7]:
from PIL import Image, ImageDraw, ImageFont

def fontImgGen(shape=(32,32), colorB=(255,255,255), colorT=(0,0,0), fontS=32, xy=(0,0), prefix='labels', numberOfImages=10):
    letters = ['!','"','#','$','%',"'",'(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[',']','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','|']
    #images = []
    sky = ImageFont.truetype('.\\fonts\\Skylark.ttf', 32)
    ubunto = ImageFont.truetype('.\\fonts\\UbuntuMono-Regular.ttf', 64)
    puppy = ImageFont.truetype('.\\fonts\\SweetPuppy.ttf', 45)
    
    chars = []
    y = []
    text = []
    i =0
    for i in range(numberOfImages):
    #for i in range(len(images)):
        if images[i]['img'].shape[0] >= shape[0] and images[i]['img'].shape[1] >= shape[1]:
            for l in letters:
                img = Image.fromarray(images[i]['img'][0:shape[0], 0:shape[1], :])
                d = ImageDraw.Draw(img)
                d.text((0, -3), l, fill=colorT, font=sky)
                chars.append(np.array(img))
                y.append(1)
                text.append(l)
                
                img = Image.fromarray(images[i]['img'][0:shape[0], 0:shape[1], :])
                d = ImageDraw.Draw(img)
                d.text((0, -23), l, fill=colorT, font=ubunto)
                chars.append(np.array(img))
                y.append(0)
                text.append(l)
                
                img = Image.fromarray(images[i]['img'][0:shape[0], 0:shape[1], :])
                d = ImageDraw.Draw(img)
                d.text((0,-22), l, fill=colorT, font=puppy)
                chars.append(np.array(img))
                y.append(2)
                text.append(l)
    return (chars, y, text)
# In[12]:
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
        #images.append({'img': img, 'words': words})
    return images
# In[36]:
images = readDataset('SynthText.h5')
images_added = readDataset('train.h5')
images.extend(images_added)
letters = ['!','"','#','$','%',"'",'(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[',']','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','|', '&', '~']
# In[39]:
chars = []
fonts = []
text = []
word = []
wordText = []
y_text = []
y = []
imgs = []
comb = []
fontClasses = ['Ubuntu Mono', 'Skylark', 'Sweet Puppy']
for i in range(len(images)):
    for w in range(len(images[i]['words'])):
        for c in range(len(images[i]['words'][w]['characters'])):
            text.append(images[i]['words'][w]['characters'][c]['char_txt'])
            y_text.append(letters.index(images[i]['words'][w]['characters'][c]['char_txt']))
            chars.append(images[i]['words'][w]['characters'][c]['resized_aligned'])
            fonts.append(images[i]['words'][w]['characters'][c]['font'])
            y.append(fontClasses.index(images[i]['words'][w]['characters'][c]['font']))
            word.append(images[i]['words'][w]['resized'])
            imgs.append(images[i]['img'])
            comb.append((i, w, c))
            #wordText.append(images[i]['words'][w]['txt'])
            
chars = np.array(chars)
y_text = np.array(y_text)
text = np.array(text)
y = np.array(y)
# In[61]:
# Generate characters augmented
(gen_chars, gen_y, gen_text) = fontImgGen(numberOfImages=10)
gen_chars = np.array(gen_chars)
#gen_chars = gen_chars.reshape(gen_chars.shape[0], gen_chars.shape[1], gen_chars.shape[2], 1)
gen_chars = random_image_augmentation(gen_chars, thetaRange=45)
gen_y = np.array(gen_y)
gen_text= np.array(gen_text)
gen_chars_gray = []
for i, c in enumerate(gen_chars):
    gen_chars_gray.append(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))
    
gen_chars = np.array(gen_chars_gray)
del gen_chars_gray
# In[ ]:

# # By character model

# The idea behind this model is basically to create a single modle per character.
# And for each model train it only on the specific character.
# 
# In the end the whole model gonna be a dictionary containing all of the characters and for each input is
# going to direct the image to needed model and predict the font accordingly.
# 
# This is add the letter of the image as a feature in our model.

models = {}
train_histories = {}
evaluation = {}
target = tf.keras.utils.to_categorical(y, num_classes=3)

rng = list(range(len(chars)))
np.random.shuffle(rng)

trainBound = int(chars.shape[0]*0.8)

chars_train = chars[rng][0:trainBound]
chars_test = chars[rng][trainBound:]
text_train = text[rng][0:trainBound]
text_test = text[rng][trainBound:]
target_train = target[rng][0:trainBound]
target_test = target[rng][trainBound:]

i=0
for let in letters:
    gen_chars_a = gen_chars[np.argwhere(gen_text==let)]
    gen_chars_a = gen_chars_a.reshape(gen_chars_a.shape[0], gen_chars_a.shape[2], gen_chars_a.shape[3])
    chars_a = chars_train[np.argwhere(text_train==let)]
    chars_a = chars_a.reshape(chars_a.shape[0], chars_a.shape[2], chars_a.shape[3])
    
    gen_y_a = gen_y[np.argwhere(gen_text==let)]
    gen_y_a = gen_y_a.reshape(-1)
    y_a = target_train[np.argwhere(text_train==let)]
    y_a = y_a.reshape(y_a.shape[0], y_a.shape[2])
    
    if chars_a.shape[0] < 2:
        test_x = chars_a
        test_y = y_a
        
        if test_y.shape[0] == 0:
            continue
        X = np.ndarray((0, 32, 32))
        Y = np.ndarray((0, 3))
    else:
        X, test_x, Y, test_y = train_test_split(chars_a, y_a, random_state=0, test_size=0.5)
        
    merged_x = np.concatenate((gen_chars_a, X))
    merged_y = np.concatenate((tf.keras.utils.to_categorical(gen_y_a, num_classes=3), Y))
    
    train_x, _, train_y, _ = train_test_split(merged_x, merged_y, random_state=0, test_size=0.0000001)
    #train_x = X
    #train_y = Y
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32,1)),
        tf.keras.layers.LayerNormalization(axis=1 , center=True , scale=True),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(88, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # history = model.fit_generator(train_generator,epochs=25,verbose = 1)
    csv_logger = tf.keras.callbacks.CSVLogger('training.csv', append=True)
    shapex = train_x.shape
    shapeT = test_x.shape
    
    train_x_norm = train_x.reshape(shapex[0], shapex[1], shapex[2], 1)/255.0
    test_x_norm = test_x.reshape(shapeT[0], shapeT[1], shapeT[2], 1)/255.0
    # train model
    print("Training model for character: {}".format(let))
    history = model.fit(train_x_norm, train_y,
              epochs=20, batch_size=64,
              validation_data=[test_x_norm, test_y],
              callbacks=[csv_logger])
    
    model.save("fontModel{}.h5".format(i))
    
    models[let] = model
    train_histories[let] = history
    
    predictions = model.predict(test_x_norm)
    
    recall = tf.keras.metrics.Recall()
    recall.update_state(test_y, predictions)
    persision = tf.keras.metrics.Precision()
    persision.update_state(test_y, predictions)
    auc = tf.keras.metrics.AUC()
    auc.update_state(test_y, predictions)
    acc = tf.keras.metrics.CategoricalAccuracy()
    acc.update_state(test_y, predictions)
    print('Accuracy: ', acc.result().numpy())
    print('Recall: ', recall.result().numpy())
    print('Persision: ', persision.result().numpy())
    print('AUC: ', auc.result().numpy())
    
    
    evaluation[let] = { 'all_samples': y_a.shape[0], 'train_samples': Y.shape[0], 'test_samples': test_y.shape[0], 'predictions': predictions, 'test_y': test_y,
                        'acc': acc, 'recall': recall, 'persision': persision, 'auc': auc }
    i +=1
# In[148]:
nsamples = 9790
total_accuracy = 0
total_percision = 0
total_recall = 0
total_auc = 0
ttt = 0
for a in evaluation.values():
    perc = a['all_samples']/nsamples
    total_accuracy += perc*a['acc'].result().numpy()
    total_percision += perc*a['persision'].result().numpy()
    total_recall += perc*a['recall'].result().numpy()
    total_auc += perc*a['auc'].result().numpy()
    ttt += a['all_samples']

print('Total Accuracy: ', total_accuracy)
print('Total Recall: ', total_recall)
print('Total Persision: ', total_percision)
print('Total AUC: ', total_auc)
# In[104]:
# # All data model + Generated Data
# Use all of the letters at once in the model to get more generality

X, test_x, Y, test_y = train_test_split(chars, tf.keras.utils.to_categorical(y), random_state=0, test_size=0.2)
# Merge the augumented data with the train data
merged_x = np.concatenate((gen_chars, X))
merged_y = np.concatenate((tf.keras.utils.to_categorical(gen_y), Y))
train_x, _, train_y, _ = train_test_split(merged_x, merged_y, random_state=0, test_size=0.0000001)
# In[42]:
# # Only given data model
train_x, test_x, train_y, test_y = train_test_split(chars, tf.keras.utils.to_categorical(y), random_state=0, test_size=0.05)
# In[43]:
# # The model
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
model.summary()
# In[44]:
# history = model.fit_generator(train_generator,epochs=25,verbose = 1)
csv_logger = tf.keras.callbacks.CSVLogger('training.csv', append=True)
shapex = train_x.shape
shapeT = test_x.shape

train_x_norm = train_x.reshape(shapex[0], shapex[1], shapex[2], 1)/255.0
test_x_norm = test_x.reshape(shapeT[0], shapeT[1], shapeT[2], 1)/255.0

earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, verbose=0, mode='max')
mcp_save = tf.keras.callbacks.ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')

# train model
history = model.fit(train_x_norm, train_y,
          epochs=20, batch_size=256,
          validation_data=[test_x_norm, test_y],
          callbacks=[earlyStopping, mcp_save, csv_logger])

# In[46]:
# # Evaluation

# Accuracy/Loss over epochs of training

reconstructed_model = tf.keras.models.load_model(".mdl_wts.hdf5")
model = reconstructed_model
#model.save("CNN_font_original+val+added.h5")
# In[64]:
import pandas as pd

def predictTestset(images_file, model, pathCSV='char_font_preds_project', test=True):
    """
    The procedure accepts dataset file in h5 format containing characters
    The function creates excel with font prediction per character
    using the following format
    ROW - IMAGE NAME - CHARACTER - FONT BINARY PREDICTION 1X3
    """
    images = readDataset(images_file, test=test)
    chars = []
    text = []
    imgs_name = []
    words = []
    y = []
    fontClasses = ['Ubuntu Mono', 'Skylark', 'Sweet Puppy']
    for i in range(len(images)):
        for w in range(len(images[i]['words'])):
            for c in range(len(images[i]['words'][w]['characters'])):
                text.append(images[i]['words'][w]['characters'][c]['char_txt'])
                chars.append(images[i]['words'][w]['characters'][c]['resized_aligned'])
                words.append(w)
                imgs_name.append(images[i]['im_name'])
                if not test:
                    y.append(fontClasses.index(images[i]['words'][w]['characters'][c]['font']))
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
        df.loc[i, vote.loc[r['image'], r['word']]]=np.max(predictions[i])
    
    df = df[['image', 'char', 'Skylark', 'Sweet Puppy', 'Ubuntu Mono']]
    df.to_csv(path_or_buf=pathCSV)
    return df, y    

# In[74]:

modelDesc = 'final_model'
results, y_val = predictTestset('SynthText_val.h5', model, test=False)
test_y = tf.keras.utils.to_categorical(y_val)
predictions = results[fontClasses].to_numpy()


# In[69]:
predictions = model.predict(test_x_norm)
modelDesc = 'final_model'

# In[26]:
import itertools

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.ylim(2.5, -0.5)
    return figure
# In[27]:
plt.figure(figsize=(12,6))
plt.subplot(121)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.subplot(122)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('LearningCurve-{}.png'.format(modelDesc))

# In[82]:
# Confusion matrix and persicion recall
pred_classes = np.argmax(predictions, axis=1)
labels = np.argmax(test_y, axis=1)
confusion_matrix = tf.math.confusion_matrix(
    labels, pred_classes, num_classes=3, weights=None, dtype=tf.dtypes.int32,
    name=None
)

fig = plot_confusion_matrix(np.array(confusion_matrix), fontClasses)
fig.savefig('ConfusionMatrix-{}.png'.format(modelDesc))
# In[80]:
# Metrics report
recall = tf.keras.metrics.Recall()
recall.update_state(test_y, predictions)
persision = tf.keras.metrics.Precision()
persision.update_state(test_y, predictions)
auc = tf.keras.metrics.AUC()
auc.update_state(test_y, predictions)
acc = tf.keras.metrics.CategoricalAccuracy()
acc.update_state(test_y, predictions)
print('Accuracy: ', acc.result().numpy())
print('Recall: ', recall.result().numpy())
print('Persision: ', persision.result().numpy())
print('AUC: ', auc.result().numpy())
# In[81]:
# ROC curve
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc

fig2 = plt.figure(figsize=(8,8))
# Plot linewidth.
lw = 2
n_classes = 3
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
fig2.show()
fig2.savefig('ROC-{}.png'.format(modelDesc))

# In[150]:
# # BoW

train = [ [[],[]] for _ in range(len(letters)) ]

# Create dense sift
sift = cv2.SIFT_create()
step_size = 4
# grid Creation assuming all characters are in the same size 32X32
kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, 32, step_size) 
                            for x in range(0, 32, step_size)]

descriptors = []
fonts = []

fontClasses = ['Ubuntu Mono', 'Skylark', 'Sweet Puppy']
for i in range(len(images)):
    for w in range(len(images[i]['txt'])):
        for c in range(len(images[i]['words'][w]['characters'])):
            (pts, descs) = sift.compute(images[i]['words'][w]['characters'][c]['resized_aligned'], kp)
            indexL = letters.index(images[i]['words'][w]['characters'][c]['char_txt'])
            fontC = fontClasses.index(images[i]['words'][w]['characters'][c]['font'])
            train[indexL][0].append(descs)
            train[indexL][1].append(fontC)
            
            descriptors.append(descs)
            fonts.append(fontC)

descriptors = np.array(descriptors)
fonts = np.array(fonts)
#chars = np.array(chars)

descriptorsFlat = descriptors.reshape(descriptors.shape[0]*descriptors.shape[1], descriptors.shape[2])


# In[151]:


from sklearn.cluster import KMeans, MiniBatchKMeans
n_clust = 100

kmeans = MiniBatchKMeans(n_clusters=n_clust, batch_size=1000, verbose=True)
get_ipython().run_line_magic('time', 'kmeans.fit(descriptorsFlat)')


# In[152]:


histograms = []

# For every decriptive image create a histogram and add to the histograms dataset
for image in descriptors:
    # predict cluster for each image descriptor
    clusters = kmeans.predict(image)
    
    #create histogram for associated clusters to descriptors
    hist, _ = np.histogram(clusters, range(n_clust+1))
    histograms.append(hist)

histograms = np.array(histograms)


# In[153]:


x_train, x_test, y_train, y_test = train_test_split(histograms, fonts)


# In[156]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

scaler = StandardScaler()
x_train_norm = scaler.fit_transform(x_train)
x_test_norm = scaler.fit_transform(x_test)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

C = 0.05
#x_train, x_test, y_train, y_test = train_test_split(X, y1, random_state=0, train_size=split, test_size=(1-split))
svc = OneVsRestClassifier(SVC(kernel='linear', C=C, probability=True, random_state=0, max_iter=1000))
svc.fit(x_train_norm, y_train_cat)
y_score = svc.decision_function(x_test_norm)


# --------------------------------------
