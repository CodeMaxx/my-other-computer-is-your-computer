# my-other-computer-is-your-computer
Microsoft Malware Classification Challenge

### Directory Structure
The git repository has the following directories -

1. **src**  - _this directory contains all the code files_
2. **feature-dump** - _this contains separate pickle files for each type of features(viz) corresponding to each malware instance's extracted features_
3. **all-features** - _this directory contains the pickle files corresponding to each malware instance's extracted features_
4. **all-feature-train** - _folder with features of train instances_
5. **all-feature-test** - _folder with features of test instances_
6. **new-files** - _folder containing file which needs to be classified. If you want to predict a class for a particular pair of .asm and .byte files, keep those files in this folder_
7. **new-files-feature-dump** - _the extracted features' pickle files are stored in this directory_
8. **new-files-all-feature-dump** - _this contains the pickle file for all features_

### Training
The training is done by running the command `python3 preprocessing.py` in the `src/` directory from the terminal. After training the trained models are stored as pickle files in the `src/` folder by their respective names. The finalModel, an object of class SupervisedModels is stored as the file `finalModels.pkl' which contains the information about scalers, features and underlying trained classifiers_

### Testing 
Testing can be done in 2 ways -

1. **Predicting the lables of test dataset** - _run the command `python3 test.py 0` in `src/` folder from the terminal. This prints out the accuracy of the model on testdata_
2. **Predicting the labels of a new file** - _run the command `python3 test.py 1 fileName` in `src/` folder from the terminal. The files `fileName.asm` and `fileName.bytes` are assumed to be in the folder `new-files/`. This prints out the predicted label by each of the underlying classifier_

Both of the testing procedures load the finalModel from the file 'finalModels.pkl' and predict the labels on the corresponding data instances.
