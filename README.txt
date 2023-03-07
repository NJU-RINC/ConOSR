This foler contains an example the training and testing code for the unknown detection experiment conducted on TinyImageNet, as described in Section 4.1 of our paper.
The running environment is python 3.6.8 & pytorch 1.8.0(alpha).

1. Training and test script
The training mechanism of the proposed method in the paper contains three steps. 
(1) Contrastive learning of the feature encoder.
(2) Training a classifier.

The python script of the step(1) is tinyimagenet_training_encoder.py, and step(2) is implemented in tinyimagenet_training_classifier.py.
Trained models are stored in the folder saved_models.
The testing script is tinyimagenet_testing.py.

2. Data organization
We do not attach the data with this code, because all the experimental datasets are widely used datasets for open access, so they can be easily obtained.

We write a data loader for loading selected classes from a dataset.
To this end, the images from each class should be placed in its corresponding folder, and the folder name should be the numeric index of its label.
For example, the training images of the first class are put in the folder './data/tinyimagenet/training/0/0'.

3. Experiments on other datasets
The experiments on other datasets, including MNIST / SVHN / CIFAR, can be conducted by changing the paths of data and model, and tuning hyper-parameters according to our description.
The data loading functions for these dataset are different, but all of them can be found in utils.py.

We also provide a test script named cifarplus_testing for CIFAR + LSUN / TinyImageNet OSR experiments.
The code of the OSR experiment on MNIST is similar to this.
The functions for different evaluation metrics are written in evaluation.py.
