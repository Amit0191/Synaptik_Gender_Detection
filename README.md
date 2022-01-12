# Synaptik_Gender_Detection
Run main.py to train the model.
Run scream.py to launch a webcam which detects your face and gender.
 You can change the input Path to image files on line 21. C:\archive\data' + '/**/*' 
 (** is for men and women folder and * is to include all files in that folder).
 Hyperparameters can be set from line 27 - 31.
 Finally, to train the model (line 128) we have used augmented(extra pics created from the given data set by rotating, scaling, etc).
 

A model with hyper parameters #2 is also included. You can directly run the file stream.py 



There are 2 plots included in the repository with hyper parameters as follows.

#1. Hyper parameters
rate = 0.0005
epoch = 100
batch_size = 64
img_dims = (128, 128, 3)



#2. Hyper parameters
rate = 0.0005
epoch = 150
batch_size = 80
img_dims = (98, 98, 3)
