## Training a CNN
 cifar10
 xavier initialization
 flip and random noise augmentation.
 avg pool 5 conv layers and 2 pool layers.
 16 filters in the first layer.
 1 FC layer with 512 nodes.
### Choice of hyperparameters:
Why I chose a 3 X 3 kernel, and stride= 2?
A kernel of size 3 is large enough to capture local spatial information and patterns in the input data, such 
as edges and corners in images. At the same time, it is small enough to keep the number of parameters 
in the network manageable, which helps to reduce the computational cost of training the network.
One of the main reasons for using a stride of 2 is to reduce the computational cost of training and 
inference. When the stride is set to 2, the feature map size is reduced by a factor of 2 in both the height 
and width dimensions, which effectively reduces the number of computations required to process the 
feature maps
### Why (0.5, 0.5, 0.5) as mean and standard deviation for normalization?
The values (0.5, 0.5, 0.5) as the mean and standard deviation to normalize the image tensor in the 
CIFAR10 dataset are commonly used because they are well-suited for the range of pixel values in the 
CIFAR10 dataset.
The CIFAR10 dataset consists of color images with pixel values in the range [0, 1], where 0 represents 
the minimum intensity (black) and 1 represents the maximum intensity (white). By normalizing the data 
with a mean of 0.5 and a standard deviation of 0.5, the transformed data will have a mean of zero and a 
standard deviation of 1
### Choosing activation function as ReLU?
I have chosen ReLU as activation function because:
Computational efficiency: ReLU activation is very simple to compute as it simply clips off the negative 
value, this makes it computationally efficient and faster to train compared to other activation functions.
Improved gradient flow: ReLU activation functions do not saturate for positive inputs, meaning that the 
gradient does not saturate and the backpropagation updates are not hindered. This leads to faster 
convergence of the model during training.
### Why learning rate lr= 0.01?
The value of 0.01 is a good compromise between the stability and speed of convergence.
### Why SGD?
SGD is a computationally efficient algorithm, as it only updates the model parameters based on a single 
training example at each iteration. This makes it well suited for large-scale datasets, where a full batch 
update would be computationally expensive.
Why momentum = 0.5 is chosen in SGD optimization?
Momentum is a technique in SGD optimization that helps to smooth out the optimization process and 
reduce oscillations. The momentum value determines the amount of influence that the current update 
has on the historical updates.
The value of 0.5 is a good compromise between sensitivity and stability
### Why CrossentropyLoss()?
I chose cross entropy loss because Cross-entropy loss, or log loss, measure the performance of a 
classification model whose output is a probability value between 0 and 1, thus it is very easy to 
interpret, and f ind accuracy

## Training an AE
 The number of AE layers will be 4 
 The classification layer will be single FC with 512 nodes
As required by the question I used the same parameters as used or training the CNN (also the choice of 
those parameters is explained above)
ConclusionIn my case CNN have performed better than AE:
 Accuracy of CNN= 54.76%
 Accuracy of AE= 10.08%
and I think the possible reason may be:
 built-in convolutional layer reduces the high dimensionality of images without losing its 
information
 Pooling layers: Pooling layers are used in CNNs to reduce the spatial size of the feature maps 
and make the network more computationally efficient.
 Weight sharing: CNNs use weight sharing, where the same set of weights is used for multiple 
positions in the input image. This makes the network more efficient, as it has fewer parameters 
to learn, and also helps to ensure that the network can generalize well to unseen images.
 Moreover the CNN architecture is specifically designed for Image processing, whereas autoencoders (AEs) are not
## References-
 https://youtu.be/nA6oEAE9IVc
 https://youtu.be/pDdP0TFzsoQ
 https://www.youtube.com/live/d9QHNkD_Pos?feature=share
 https://youtu.be/wnK3uWv_WkU
 https://youtu.be/9aYuQmMJvjA
 https://youtu.be/wcQuJOZedlE
 https://youtu.be/zp8clK9yCro
 Data Augmentation Explanation (openai.com)
