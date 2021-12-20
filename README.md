# Font recognition project
The Objective: given a dataset of N pictures containing Arbitrary scenery overlayed
by different textual data, classify each letter in the dataset to the correct font.
Few presumptions on the data:
* Each overlayed word is written in one font
* There are 3 fonts to classify
* The words can be rotated, variated in color, opacity, shear and size.
* A word bounding box is given
* A character bounding box is given 

![image](https://github.com/nitzan-karni/fontRecognition/blob/main/bay%2Barea_90_bb.jpg)

Implementation: 

Corped the characters by the given bounding boxes and splitted to different images.
Converted the color to grey on each character.
Resized to image size of 32X32.
Streighten and align the charcters to standard orientation.

![image](https://user-images.githubusercontent.com/51075167/114318713-f8c24880-9b16-11eb-9bea-c0ffe771a62c.png)

Eventually after multiple approaches ( data synthesis, by character ensemble model, VGG16 / Mobilenet)
I decided the best train efficiancy and genralization to accurracy trade off is offered by the simplest CNN

* conv2d_20 (Conv2D)           (None, 30, 30, 32)        320       
* layer_normalization_8 (Layer (None, 30, 30, 32)        60        
* max_pooling2d_20 (MaxPooling (None, 15, 15, 32)        0         
* conv2d_21 (Conv2D)           (None, 13, 13, 64)        18496     
* max_pooling2d_21 (MaxPooling (None, 6, 6, 64)          0         
* flatten_9 (Flatten)          (None, 2304)              0         
* dense_20 (Dense)             (None, 88)                202840    
* dense_21 (Dense)             (None, 3)                 267       

Total params: 221,983
Trainable params: 221,983

loss: mean_squared_error
Optimizer: adam

Training data: 9790 samples
Validation data: 2448 samples
Epochs: 20
Time to train: 7min

And after a certain letter is predicted I set a vote between the other letters in the word which deciedes on the word's font.

## Results
* Accuracy:  0.96377164
* Recall:  0.95401317
* Precision:  0.96627134
* AUC:  0.9784279


![image](https://user-images.githubusercontent.com/51075167/114319858-f2829b00-9b1b-11eb-9535-449461af06fd.png)

### before voting
![image](https://user-images.githubusercontent.com/51075167/114319867-f9111280-9b1b-11eb-9fa8-92e044b47033.png)
![image](https://user-images.githubusercontent.com/51075167/114319879-00382080-9b1c-11eb-976e-91e5c39d94db.png)

## runing the model on the test set

* choose a path to dataset to predict. ( f.e test.h5, if the dataset is in the same directory )
* choose the name  of the output file, must end with .csv ( f.e test_predicted.csv )
* run the following code from the commandline, at the same directory of the model

```
python test_predict.py -set test.h5 -outfile test_predicted.csv
```

the following code should create the predictions csv file.

---

## training the model
* The code to train the model appears in `train_model.py` file.
* If you wish to retrain the model you can run the code again using.
  just move the `train.h5`, `SynthText.h5`, `SynthText_val.h5` files to the model directory
```
python train_model.py
```

---
## added files
The following files are added for further invistigation:

* `test_predicted.csv` - predicted fonts on given test dataset
* `finalModel.h5` 	 - pretrained model.
* `test_predict.py`	 - The file to execute preprocess and predictions on given dataset.
* `test_predict.ipynb`	 - The same as `test_predict.py` just in notebook.
* `train_model.py` 	 - The code used to train the final model.
* `project.ipynb` 	 - jupiter notebook containing the research process I conducted. contains the vizualizations, evaluations and failed models.
* `project_process.py` - same as `project.ipynb` but in as py file. a little more orginized.


