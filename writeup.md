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


I provide a bar chart below showing the class/label data percentage over the data between the three dataset splits of training, validation and test.  This chart verifies that the data splits chosen are similar to each other in class proportions.  However, we can see that the classes are highly imbalanced.  Classes such as 0 (Speed limit (20km/h)), 19 (Dangerous curve to the left), and 24 (Road narrows on the right) have less than 20% as populous as more common signs such as 13 (Yield).  This makes it far harder for the network to generalize and correctly predict these rarer classes.  A useful strategy for a further step would be to balance the sampling of sign classes or further augment the rare sign classes to equalize the class proportions.  I have chosen not to do any data augmentation or balancing at this time.

![alt text][image10]

Further results and visualization can be found in my notebook code at [project code] (https://github.com/dataronio/CarND-Traffic_Sign_Classifier-P2/blob/master/Traffic_Sign_Classifier.ipynb).

### Design and Test a Model Architecture ###


