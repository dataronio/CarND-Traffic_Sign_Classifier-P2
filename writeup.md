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

I provide a bar chart below showing the class/label data percentage over the data between the three dataset splits of training, validation and test.  This chart verifies that the data splits chosen are similar to each other in class proportions.  However, we can see that the classes are highly imbalanced.  Classes such as 0 (Speed limit (20km/h)), 19 (Dangerous curve to the left), and 24 (Road narrows on the right) have less than 20% as populous as more common signs such as 13 (Yield).  This makes it far harder for the network to generalize and correctly predict these rarer classes.  A useful strategy for a further step would be to balance the sampling of sign classes or further augment the rare sign classes to equalize the class proportions.  I have chosen not to do any data augmentation or balancing at this time.

![alt text][image10]

Further results and visualization can be found in my notebook code at [project code](https://github.com/dataronio/CarND-Traffic_Sign_Classifier-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Design and Test a Model Architecture ###

#### Pre-process the Data Set (normalization, grayscale, etc.) ####

Previous researchers have found that grayscaling the color images does little damage to accuracy and is easier to train.  Normalization of neural network inputs to a small range such as [-1,1] centered about zeros improves the scaling of the weights and thus inproves the convergence of gradient descent.  Following these best practices I grayscale the images using standard OpenCV functions and normalize be simply subtracting by 128.0 followed by dividing the result by 128.0.  This normalizes image data which is in the range [0, 255] into the range [-1, 1].

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




 


