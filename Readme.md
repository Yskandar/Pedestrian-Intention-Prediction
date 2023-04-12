# Project Description
This project focuses on a common problem in the autonomous driving industry: Pedestrian Intention Prediction.
We are given a set of video clips along with some metadata about the pedestrians on each of those clipcs and their stances. The problem we propose to solve is to predict for each pedestrian whether or not they are crossing the street, and whether or not they will be crossing the street at the next state.

The dataset from which the clips are issued is the opensource JAAD dataset, which you can read more about here: http://data.nvision2.eecs.yorku.ca/JAAD_dataset/ if you are interested.

The code and results are available in the [jupyter notebook file](PA_2021_CC_C.ipynb) in the root folder.
# My take on this problem
This task is a classification problem. We want to classify each image along with the vector of given features (moving_slow, stopped, handwave...) as 'crossing' or 'not crossing'.

## 1 : Processing the data 

One of the key aspects here is that time matters: using images independently from one another would surely result in bad classification accuracy. Thus, I am not sure how a 'per pedestrian frame' dataframe would be of use here.

The first objective is to regroup, for each pedestrian, the corresponding sequences of cropped images and features. However, since we need to keep the order of the images, we cannot simply regroup all the images of each pedestrian together.

What we can do, is regroup images corresponding to the same trajectory, regardless of the pedestrian. This means that we use the pedestrian ID just to constitute a true trajectory, not to regroup all the trajectories together per pedestrian. 
Thus, for each row of the dataframe, we need to constitute a trajectory (sequence of images and corresponding features) that would constitue one sample for the training of our model.

I tried processing images, retrieveing the parts corresponding to the bounding boxes and storing them. The main issue was the following: to feed a convolutional networkthe images need to be all of the same shape, thus they need to be resized, and this operation takes a lot of time. Considering the total number of frames (128220), processing all these frames would have taken hours and a lot of memory. Thus, to avoid this issue, I had to build a dataloader that would create batches of images, crop them according to the bounding boxes and feed them to the networks.

To build this dataloader, several steps were required:

    * retrieve images using the frame indices and the video ID
    
    * crop images according to the bounding boxes, resize them
    
    * convert the feature vector to binary values (0: False, 1: True), retrieve the labels
    
    * create an object, containing the sequence of images and the sequence of feature vectors
    
    * regroup trajectories into training and validation datasets
    
## 2 : Building models

The next step is to build models. When time is involved, two types of neural networks are considered: CNNs and RNNs. Furthermore, since we are processing two types of data at the same time (images and feature vectors), we need to use two networks in parallel and figure out a way to output a prediction based on the outputs of these two networks. We need to perform the following tasks:

    * Build a network to process the images
    
    * Build a network to process the feature vectors
    
    * Output the prediction from the output of the two networks
    
    * Train the networks accordingly, see how they perform

For this task, I decided to divide all the existing trajectories into samples of [trajectory_length] timesteps.
Thus, I am feeding my networks with the following data:

    * N batches of samples of the shape (batch_size, trajectory_length, image_size) for the network processing images
    * N batches of samples of the shape (batch_size, trajectory_length, number of features) for the network processing the metadata


# Results

The code and results are available in the [jupyter notebook file](PA_2021_CC_C.ipynb) in the root folder.

## 1. Model performances



### 1. **Models**
Overall, I was not able to get meaningful feature extraction using the CNN network. I attribute this mostly to the complicate task of combining LSTMs and CNNs and the image processing step that resizes images and thus drops information. However, we can see clearly that the training and validation losses are going down, which is a good thing. Sadly, this gain does not seem enough for an improvement in accuracy.

The feature network performs well, and manages to achieve good training/validation accuracies (98%). This is mostly due to the fact that a lot of meaningful features are given, and the combination of features such as 'clear path', 'moving_slow' or 'stopping' gives a pretty good hunch on wether or not the pedestrian is crossing.

The combination of the two networks doesn't perform as well as the feature network alone, which is mainly due to the CNN-LSTM not being able to provide meaningful predictions. One could argue that adding the predictions from the CNN-LSTM with the predictions from the feature network is not the smartest way to combine the predictions of both networks.


### 2. **Weakness of my method**
The main weakness of the method I used in this notebook is that it requires the use of groups of frames that constitute a trajectory. This means that you need to feed the network with a group of images corresponding to the same trajectory, and cannot just give one image alone. 

Due to the linear layers of the CNN-LSTM network, we use all the images of the given trajectory to predict the labels, meaning that for each image we use information of every other image to predict a label, which is the opposite of what was advocated in the beginning. To circumvent this, one could simply remove the linear layers and just change the last lstm layer to fit the shape (trajectory_length, 1), however, this technique gives worst results.



## 2. Ways to improve predictions


### 1. **Image processing**
The first axis of improvement is the image processing step. I don't know how much of the information is lost when images are resized, nor do I know the most appropriate image size to use for the CNN.


### 2. **Class imbalance:'crossing' frames are outnumbered**
It seems that the dataset is uneven regarding the number of 'crossing' and not crossing frames. Thus, I suspect the feature network to output '0' most of the time and get a good accuracy due to the high number of 'not crossing' frames in there. It would be judicious to put weights on the classes to remedy this issue


### 3. **Dimensionality reduction**
One of the problem of the given dataset is that it presents a huge number of frames. Thus, it is likely that there is a redundancy in the data. This is a problem that often affects big datasets, and there are several ways to handle this 'diminishing returns' issue. One of the way to circumvent this issue is to train only on the most relevant datasamples. This can be achieved by selecting the samples that contribute the most to the gradient when doing the gradient descent step. See  [CRAIG](https://arxiv.org/pdf/1906.01827.pdf) for an example.


### 4. **Image segmentation**
Another way to reduce the size of the datasets would be to perform segmentation on the input images to keep only the body of the pedestrians. Such task could be performed using a U-net trained for body segmentation. After predicting the masks for our input images, we would apply these masks to our input images and feed them to the CNN. This would probably help feature extraction


### 5. **Optimizing the trajectory length**
For this study, I decided to use a fixed trajectory length of 4. However, a deeper study on the appropriate trajectory length would give us better results.


### 6. **Getting the position/velocity of the pedestrians in the space**
If we had access to the position of the pedestrians in the image and their velocity (which could be computed using classical image processing techniques: optical flow...), we could probably achieve a better accuracy, since this information matters a lot in the prediction process. 




# References

1. Amir Rasouli, Iuliia Kotseruba and John K. Tsotsos: Are They Going to Cross? A Benchmark Dataset and Baseline for Pedestrian Crosswalk Behavior. In Proceedings of the IEEE International Conference on Computer Vision Workshops, pages 206-213, 2017.

2. Amir Rasouli, Iuliia Kotseruba and John K. Tsotsos: Agreeing to cross: How drivers and pedestrians communicate. In IEEE Intelligent Vehicles Symposium (IV), pages 264-269, 2017.

3. Baharan Mirzasoleiman, Jeff Bilmes and Jure Leskovec: Coresets for Data-efficient Training of Machine Learning Models. In Proceedings of the 37 th International Conference on Machine Learning, Vienna, Austria, PMLR 119, 2020.

