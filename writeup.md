# Traffic Sign Recognition Project #


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscaling"
[image2]: ./examples/roadwork.jpg "Roadwork"
[image3]: ./examples/no-limit.jpg "End of all speed limits"
[image4]: ./examples/no_passing.jpg "No Passing"
[image5]: ./examples/priority.jpg "Priority Road"
[image6]: ./examples/bump.jpg "Bumpy Road"
[image7]: ./examples/traincounts.jpg "Train Counts"
[image8]: ./examples/valcounts.jpg "Validation Counts"
[image9]: ./examples/testcounts.jpg "Test Counts"
[image10]: ./examples/relcounts.jpg "Relative Counts"


## Writeup on the Traffic Sign Recognizer Project ##

### Dataset Summary & Exploration ###

I used the Numpy library to create summary statistics for the German Traffic Signs Dataset:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Traffic sign image shape is in the form (Height, Width, Channels).  The dataset is in color and thus occupies 3 channels of red, green and blue.

### Exploratory Visualization of the Dataset ###

A file encoding class label with sign type is [signnames.csv](https://github.com/dataronio/CarND-Traffic_Sign_Classifier-P2/blob/master/signnames.csv)

I provide a bar chart below showing the class/label data percentage over the data between the three dataset splits of training, validation and test.  This chart verifies that the data splits chosen are similar to each other in class proportions.  However, we can see that the classes are highly imbalanced.  Classes such as 0 (Speed limit (20km/h)), 19 (Dangerous curve to the left), and 24 (Road narrows on the right) are less than 20% as populous as more common signs such as 13 (Yield).  This makes it far harder for the network to generalize and correctly predict these rarer classes.  A useful strategy for a further step would be to balance the sampling of sign classes or further augment the rare sign classes to equalize the class proportions.  I have chosen not to do any data augmentation or balancing at this time.

![alt text][image10]

Further results and visualization can be found in my notebook code at [project code](https://github.com/dataronio/CarND-Traffic_Sign_Classifier-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Design and Test a Model Architecture ###

#### Pre-process the Data Set (normalization, grayscale, etc.) ####

Previous researchers have found that grayscaling the color images does little damage to accuracy and is easier to train.  Normalization of neural network inputs to a small range such as [-1,1] centered about zeros improves the scaling of the weights and thus inproves the convergence of gradient descent.  Following these best practices, I then grayscale the images using standard OpenCV functions and normalize be simply subtracting by 128.0 followed by dividing the result by 128.0.  This normalizes image data which is in the range [0, 255] into the range [-1, 1].

Here is an example of grayscaling an image in this dataset:

![alt text][image1]

#### Model Architecture ####

Detail of my final model architecture follows:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32   |
| RELU                  |                                               |
| Max pooling 2x2       | 2x2 stride,  outputs 5x5x32                   |
| Flatten               | output 800                                    | 
| Fully connected		| output 120        							|
| RELU                  |                                               |
| Dropout               | probability 0.5                               |
| Fully connected       | output 80                                     |
| RELU                  |                                               |
| Dropout               | probability 0.5                               |
| Softmax				| 43 classes        							|
|						|												|

#### Training the Model ####

The objective for the model is softmax cross-entropy using the Adam optimizer option in Tensorflow.
The network weights were initialized as truncated normal (mean=0, sigma=0.1). Biases were always initialized at zero.  The learning rate was 0.001 and
I coded a simple early stopping code to save the model after an increase in validation set accuracy.
The model was then trained for 200 epochs with a 128 batch size. I found a smaller batch size was useful in adding noise which kept the models from plateauing during training.  Higher batch sizes appeared more stable during training but would often get stuck before reaching full potential on the validation set.

#### Approach for Finding Final Model Architecture ####

I initially used a model quite similar to the original LeNet model just to get things working.  LeNet style convolutional neural networks have been show to have excellent results on image data much like the current traffic sign dataset.  This model had two convolutional-maxpool layers followed by two fully connected layers that were each followed by dropout(Prob=0.5) to help generalize better.  LeNet has only a single dense layer before the softmax output.  With this first network, I couldn't achieve results reliably above 0.92 accuracy.  I then decided to increase the size of the network to combat this perceived underfitting.  My second model contained 3 convolutional-MaxPool layers (6, 32, 64 filters) followed by two large fully-connected (1500, 500 neurons).  I found this model interesting but much more difficult to train.  Further work on a model of this size could probably lead to useful results after balancing and augmentation of the dataset. Increases of the learning rate caused instability during training.  Further work on annealing down the learning rate as training progresses would probably be useful as well.

I decided to use a slightly larger version of my first model for the final architecture.  Training was stable and progressed nicely.  Dropout appeared to help validation accuracy immensely.  Randomly dropping neurons in the final layers helped to keep the network from overfitting and helped to create more stable models on the validation set.

After working on the third and final architecture and being happy with validation results, I ran the test data and ended all experimentation.

Final model results were:

* validation set accuracy of 0.96  
* test set accuracy of 0.943

Test set accuracy holds up quite nicely.  I do not feel that overfit is a major issue.

### Test the Model on New Images ###

#### Discussion of New Images ####

I found five images of German traffic signs on the web to use as a new data test:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

The first image (roadwork) is distinctive and fills up the frame nicely.  The sign itself is more weather beaten than typical images in the dataset.

The second image (end of all speed limits) is a rare sign that should be a difficult test for the network.  The fine lines of this sign will be difficult to resolve with only a 32x32 image.  It is also small in relation to the image.  Most images in the dataset have a very small margin around them.  This should be a difficult test for the network.

The third image (no passing) is in bad condition and should be difficult to resolve.  The no passing sign (red car on left) is an example that shows that using color in the input rather that simply grayscale could be helpful information.

The fourth image (priority road) is in excellent shape and is a very common sign in the dataset.  I don't expect any issues with this sign.

The final image (bumpy road) is in used condition on a completely black background.  This is considerably different than the typical sign background in the dataset and should prove a challenge.

#### Discussion of model predictions on New Signs ####

New sign prediction accuracy is only 0.4 (2 of 5 signs).  Accuracy is greatly reduced from test set accuracy.  In my discussion above,  I highlight the differences of my web images with the dataset.  My web images in general are smaller in the frame, in much worse condition and some have different backgrounds than images from the dataset.  The German Traffic Sign dataset has certain requirements for the margin size and this is not met by my images.     

The predictions of the network is as follows:

| Input Image     		|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road Work (25)      	|  Road Work (25) - correct   					| 
| End of all Limits (32)| Right of way at next intersection  (11)     	|
| No Passing (9)        | End of No Passing (41)                        |
| Priority Road (12)    | Priority Road (12) - correct                  |
| Bumpy Road (22)       | Speed Limit (20 Km/hr)                        |
|                       |                                               |


#### Analyze the Softmax Probability Predictions for New Signs ####

For the first image, the model is completely sure that this is a road work sign (probability of ~1.0), and the image is a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0         			| Speed Limit (60 Km/hr) 						| 
| .0    				| Speed Limit (50 Km/hr) 						|
| .0					| Speed Limit (30 Km/hr) 						|
| .0	      			| Speed Limit (20 Km/hr) 						|
| 1.0				    | Road Work         							|


For the second image, the model is completely sure that this is a right-of-Way sign (probability of ~1.0), but the image is a end of all speed limits sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0         			| Speed Limit (60 Km/hr) 						| 
| .0    				| Speed Limit (50 Km/hr) 						|
| .0					| Speed Limit (30 Km/hr) 						|
| .0	      			| Speed Limit (20 Km/hr) 						|
| 1.0				    | Right-of-Way at Next Intersection     		|

For the third image, the model is completely sure that this is a end of no passing sign (probability of ~1.0), but the image is a no passing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0         			| Speed Limit (60 Km/hr) 						| 
| .0    				| Speed Limit (50 Km/hr) 						|
| .0					| Speed Limit (30 Km/hr) 						|
| .0	      			| Speed Limit (20 Km/hr) 						|
| 1.0				    | End of No Passing    							|


For the fourth image, the model is completely sure that this is a priority road sign (probability of ~1.0). The image has been correctly identified. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0         			| Speed Limit (60 Km/hr) 						| 
| .0    				| Speed Limit (50 Km/hr) 						|
| .0					| Speed Limit (30 Km/hr) 						|
| .0	      			| Speed Limit (20 Km/hr) 						|
| 1.0				    | Priority Road       							|


For the final image, the model is completely sure that this is a Speed Limit (20 km/hr) (probability of ~1.0), but the image is a bumpy road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0         			| Speed Limit (60 Km/hr) 						| 
| .0    				| Speed Limit (50 Km/hr) 						|
| .0					| Speed Limit (30 Km/hr) 						|
| .0	      			| Speed Limit (20 Km/hr) 						|
| 1.0				    | Bumpy Road         							|

The output probabilities are saturated to 1.0 for each test image.  This is not a good sign and seems to indicate overfit even though it gets excellent out of sample (test) result on the dataset.  Alternate forms of German Traffic Signs such as my web images are not well classified and the network is always far too sure of its classification.  My prescription for these issues in a further optional investigation is to introduce impulse noise (speckle) into the dataset to help regularize and improve generalization.  Also, as I mentioned previously augmentation and balancing the dataset are indicated as well.







 


