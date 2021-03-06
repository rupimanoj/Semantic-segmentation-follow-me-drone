[network_arch]: ./data/FCN_netwok_arch.PNG
[network_arch]: ./data/chair.PNG
[graph_low_epoch]: ./data/graph_low_epoch.PNG
[graph_low_rate]: ./data/graph_slow_learn.PNG
[graph_overfitting]: ./data/validation_graph.PNG
[graph_solution]: ./data/solution_graph.PNG
[result_low_epoch]: ./data/result_low_epoch.PNG
[result_low_rate]: ./data/result_slow_learn.PNG
[result_overfitting]: ./data/result_overfitting.PNG
[result_solution]: ./data/solution_result.PNG
[chair]: ./data/chair.PNG
[high_level_features]: ./data/high_level.PNG
[skip_connections]: ./data/skip_connection.PNG
[upsampling]: ./data/upsampling.PNG
[hero_slow_learn]: ./data/hero_slow_learn.PNG
[hero_low_epoch]: ./data/hero_low_epoch.PNG
[others_slow_learn]: ./data/others_slow_learn.PNG
[others_low_epoch]: ./data/others_low_epoch.PNG
[conv_exp]: ./data/conv_exp.PNG

### Network Architecture:

To perform semantic segmentation, fully convolution network is used. <br/>
Initially reduced spatial features of image are captured using FCN approach. Using FCN, downsized feature map is achieved, where spatial information is retained. Thereafter, Decoder block is used to achieve pixel level classification. In decoder flow, up sampling techniques were used to reconstruct the image. Concatenation techniques were used to combine encoder layers to each layer in the decoder block to achieve finer level pixelwise segmentation. <br/><br/><br/>

Important details and clear explanation of network are explained below.

#### Why 1x1 convolution and FCN?


As part of earlier lab exercises fully connected convolution networks are used to classify images. In fully connected convolution networks, at final stages of network, spatial information of features is lost as all the pixel values in reduced feature maps are spread vertically (no significance is given to pixel location) to make connections to neurons in next layer. However, as the task for semantic segmentation is to make classification at pixel level, the retention of spatial knowledge of collected features becomes important. To solve this problem we have been introduced to the concept of 1x1 convolution and FCN (Fully convolution networks). As in 1x1 convolution, operations are performed only at single pixel level and adjacent pixels will not have any impact in deciding the value of corresponding pixel in next layer feature map, spatial information is kept intact. All the calculations and dimensionality reduction of feature maps will happen only along the depth direction. <br/><br/><br/>



![alt text][network_arch] <br/>

### Network architecture code:

``` python
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder1 = separable_conv2d_batchnorm(inputs, 32, strides=2)
    encoder2 = separable_conv2d_batchnorm(encoder1, 64, strides=2)
    encoder3 = separable_conv2d_batchnorm(encoder2, 64, strides=2)
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    fcn_encoded = conv2d_batchnorm(encoder3, 64, kernel_size=1, strides=1)
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decode1 = decoder_block(fcn_encoded,encoder2,64)
    decode2 = decoder_block(decode1,encoder1,64)
    x = decoder_block(decode2,inputs,32)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)
```

#### Few important considerations and insights:

* In encoder layer, with stride of 2 at each layer, spatial size of image is reduced from 160 -> 80 -> 40 ->20
* In decoder layer, with up sampling of factor 2 is used. Hence feature map size is increased from 20 -> 40 ->80 ->160.
* While concatenating layers from encoder layer to decoder layer as part of skip connections, care is taken that feature map of equal spatial size are concatenated.


### Parameters selected:

For hyper tuning parameters, I have employed brute force approach. Batch size of 64 is fixed as too small size may result in frequent updates of weights and large size requires more memory. Once batch size is fixed, steps_per_epoch and validation_steps can be calculated from available data set size.
The important parameters learning_rate and no_of_epoch played a significant role in achieving final IOU metric. Combination of learning rates from 0.005, 0.01 to 0.1 and no_of_epochs from 10,15 to 20 are used. It is observed that best results are obtained for combination of learning rate of 0.05 and 15 number of epochs.

``` python
learning_rate = 0.01
batch_size = 64
num_epochs = 15
steps_per_epoch = 140
validation_steps = 30
workers = 2
```
#### Explanation about hyper parameter tuning.
<br/>

As the underlying error optimization function uses stochastic batch gradient descent function, the significance of learning rate and number of epochs can be similarly related to that of simple continuous regression learning. For stochastic gradient descent optimization, few thumb rules, that can be remembered are pointed below. <br/>

* Keep training network till the error function reaches the vicinity of minimum error value. Low number of epochs give less number of chances to reach minimum error value. However, on other side with high number of epochs we have risk of overfitting, as with each repetition model is getting tied to training data and is no more a generalized model. Such overfitting cases can be identified using validation error. <br/>

* Similarly, low training rate will take more number of epochs to reach minimum value. More epochs mean more learning time and compute power. However, with higher learning rate we have risk of overjumping the minimum error value. <br/>
To address learning rate tuning issue, techniques have been proposed to use variable leaning rate instead of single learning rate. Broad idea behind this approach is to use higher learning rates initially and as network approaches minimum value decrease the learning rate. Apart from varying learning rate with time, we also have techniques to vary the learning rate for each weight parameter. <br/>
In follow_me deep learning project we have taken this approach to tune the learning rate parameter. This can be achieved by <b>adagrad optimizer</b>. Even though adagrad optimizer tunes the learning rate intelligently with iterations, the initial learning rate supplied to the optimizer plays critical role in the rate at which minimum error value is obtained. <br/>

The above insights are explained with  below results from the project. <br/>

| Case        | Learning rate           | epochs  | IOU  |
| ------------- |:-------------:| -----:| -----:|
| case1  (low epoch)    | 0.01 | 5 | 28.2|
| case2   (low learn)    | 0.005      |   5 | 24.6 |
| case3(solution case)  | 0.01     |    15 | 42.3 |
| case4 (over fitting) | 0.01      |    20 | 36.9 |

#### Case 1:<br/>

![alt text][graph_low_epoch] <br/>
![alt text][result_low_epoch] <br/>
<b>Comments:"</b> Required IOU metric is not obtained. Loss value is in decreasing trend and not yet saturated. Can use more epoch to minimize the loss value and improve IOU metric.

#### Case 2:<br/>
![alt text][graph_low_rate] <br/>
![alt text][result_low_rate] <br/><br/>

<b>Comments:"</b> Required IOU metric is not obtained. Even though number of epochs are same, IOU value is decreased, this can be attributed to low learning rate. Minimum loss value is not yet obtained. Increase number of epochs and learning rate.

#### Case 3:<br/>
![alt text][graph_solution] <br/>
![alt text][result_solution] <br/><br/>

<b>Comments:"</b> Required IOU metric is obtained. Validation error and loss value are closely related and  there are no obvious signs of overfitting.

#### Case 4:
![alt text][graph_overfitting] <br/>
![alt text][result_overfitting] <br/><br/>

<b>Comments:"</b> Required IOU metric is not obtained even though number of epochs increased. We can see a sharp rise in validation error. This can be attributed to overfitting.

#### Encoder block:

Encoder block is used to capture higher level hidden features of an image. With each layer in an encoder, the complexity of hidden features getting capture will be increased. For example, first layer will be used to capture edges and corners in an image, second hidden layer can be used to capture circular blobs, parallel lines or 'T' shape in an image and next layer will be used to capture more complex structures such as honeycomb like sructure. In training process, convolution filters are learned to capture this latent feature in an image. <br/>

For example, in below image even though chairs look different in color and size, its hidden features are same. Parallel lines along legs, number of plane surfaces are same in both chairs. Encoder block is trained to capture this high-level features and treat both chairs belong to same class. <br/>
![alt text][chair] <br/>

Below images show some higher level features encoder block tries to capture from an image. <br/>

![alt text][high_level_features] <br/>
Below image shows activation map of convolution filters in hidden layers that are used to capture parallel lines, diagonal like structures, circular blobs etc.<br/>
![alt text][conv_exp] <br/>

#### Separable convolutions:

In encoder block, for the initial hidden layers, separable convolution 2d is used instead of normal convolution procedure. As explained in tutorial, separable convolution has an advantage to maintain less number of weights than compared to normal convolution procedure. In theory, this can be shown by claiming convolving an image with two one dimensional matrices in sequence can be equivalent to convolving an image with single 2-dimensional matrix. <br/><br/>

H = H1 x H2, H is a 2-dimensional matrix with size axb, H1 with ax1, H2 with 1xb. It can be shown that axb is significantly larger than a+b. With this argument, we can say that separable convolutions need less number of weights to learn. For details please refer lecture slides. <br/>

``` python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```
<br/><br/>

#### Decoder block:

By the end of encoder block, the latent representation of an image will capture only hidden features of an image, but pixel level information is lost. To reconstruct the pixel level information from latent representation/condensed representation we need to use interpolation techniques. For this project, we used bilinear up sampling is used. Mathematically, bilinear up sampling uses linear combination of adjacent samples to construct next layer image. Nearby samples will have more say than faraway samples. <br/>

![alt text][upsampling] <br/>

#### skip connections.

Even though we try to reconstruct an image from its condensed form/latent representation through interpolation, latent representation will have information only about local features. A global level information of an image is lost in latent representation. To gain this information, we use skip connections rom encoder block to decoder block. Skip connections will be required for fine grained labelling of pixels in semantic segmentation. <br/>

Below picture illustrates the significance of skip connections. <br/>

![alt text][skip_connections] <br/>

For decoder block, at each layer three important steps a performed as stated in notebook. For reference including those steps. <br/>

* A bilinear up sampling layer using the upsample_bilinear() function. The current recommended factor for up sampling is set to 2.
* A layer concatenation step. This step is like skip connections. You will concatenate the up sampled small_ip_layer and the large_ip_layer.
* Some (one or two) additional separable convolution layers to extract some more spatial information from prior layers.

``` python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    output_layer = BilinearUpSampling2D((2,2))(small_ip_layer)
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    output_concatenate = layers.concatenate([output_layer, large_ip_layer])
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(output_concatenate, filters, strides=1)
    return output_layer 
```

<br/><br/>

#### Extending model to other classes such as dogs and cats:

With existing model, weights and data, it will become erroneous to make pixel wise classification for other categories such as cat or dog. We will be requiring data of other classes to train the network. However, new techniques have been proposed to do transfer learning for semantic segmentation task too. Such techniques are beyond scope of this course. An example can be found in this publication. (https://pdfs.semanticscholar.org/1837/decb49fb6fc68a6085e797faefb591fecb8a.pdf)


### Future work

* Collect more data related to special cases. One case being hero is far from the camera and is in the crowd. IOU metric is affected significantly by these cases. 
* Learn techniques and methods to understand the internal representation of network. This will help to tune the parameters more systematically than doing it by brute force approach.
