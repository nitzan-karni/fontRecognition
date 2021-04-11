# Font recognition project
![image](https://user-images.githubusercontent.com/51075167/114317977-63718500-9b13-11eb-89a4-d8130728ca92.png)

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


