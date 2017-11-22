[network_arch]: ./data/FCN_netwok_arch.PNG

### Network Architecture:

To perform semantic segmentation, fully convolution network is used. <br/>
Initially reduced spatial features of image are captured using FCN approach. Using FCN downsized feature map is achieved , where spatial information is retained. Thereafter, Decoder block is used to achieve pixel level classification. In decoder flow, up sampling techniques were used. Concatenation techniques were used to combine encoder layers to each layer in decoder block to achieve finer level pixelwise segmentation. <br/><br/><br/>

Important details and clear explaination of network are explained below.

#### Why 1x1 convolution and FCN?


As part of earlier lab exercises fully connected convolution networks are used to classiy images. In fully connected convolution networks, at final stages of network, spatial inormation of features is lost as all the pixel values in reduced feature maps are spread vertically (no significance is given to pixel location) to make connections to neurons in next layer  . However as the task for semantic segmentation is to make classification at pixel level, the spatial importance of features becomes important. To solve this problem we have been introduced to concept of 1x1 convolution and FCN(Fully convolution networks). As in 1x1 convolution, operations are perormed only at single pixel level and adjacent pixels will not have any impact  in deciding the value of corresponding pixel in next layer fetaure map, spatiial infomation is kept intact. All the calculations and dimensionality reduction of feature maps will happen only along the depth direction. <br/><br/><br/>

#### Encoder block:

#### Seprable convolutions:

In encoder block, for the initial hidden layers, seperable convolution 2d is used instead of normal convolution procedure. As explained in tutorial, seperable convolution has  an advantage to maintain less number of weights than compareed to normal convolution procedre. In theory, this can be shown by claiming convolving an image with two one dimensional matrices in sequence can be equivlent to cnvolving an image with single 2-dimensional matrix. <br/><br/>

H = H1 x H2, H is a 2 dimensional matrix with size axb, H1 with ax1, H2 with 1xb. It can be shown that axb is significaantly larger than a+b. With this argument we can say that seprable convolutions need less number of weights to learn. For details please refer lecture slides.<br/>

``` python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```
<br/><br/>
#### Decoder block: Upsampling, concatenation, spatial information extraction

For decoderr block, at each layer three important steps a performed as stated in notebook. For reefernce including  those steps. <br/>

* A bilinear upsampling layer using the upsample_bilinear() function. The current recommended factor for upsampling is set to 2.
* A layer concatenation step. This step is similar to skip connections. You will concatenate the upsampled small_ip_layer and the large_ip_layer.
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

* In encoder layer, wth stridde of 2 at each layer, spatial size of image is reduced from 160 -> 80 -> 40 ->20
* In decoder layer, with upsampling of factor 2 is used. Hence feature map size is increased from 20 -> 40 ->80 ->160.
* While concatenationg layers from encoder layer to decoder layer, care is taken that feature mapof equal spatial ssize ar concatenated.
* In decoder layer also we sue, seprable convolutional layers,to learn weights that extract spatial informaion from prior layers and finally attain pixel level segmentation in laast layer.<br/>

#### Extending model to other classes such as dogs and cats:

With existing model, weights and data, it will become erreneous to make pixel wise classification for other categories such as cat or dog. We will be requiring data of other classes to train the network. However, new techniques have been proposed to do transfer learning for semantic segmentation task too. Such techniues are beyond scope of this course. An example can be found in this publication. (https://pdfs.semanticscholar.org/1837/decb49fb6fc68a6085e797faefb591fecb8a.pdf)

### Parameters selected:

For hypertuning parameters, I haave employed brute force approach. Batch size of 64 is fixed as too small size may result in frequent updates of weights and large size reqires more memory. Once batch size is fixed, steps_per_epoch and validation_steps can be calulated from avalible data set size.
The impotant parametes learning_rate and no_of_epoch played a significant role in achieving final IOU mettric. Combination of earnig weights from 0.005, 0.01 to 0.1 and no_of_epochs from 10,15 to 20 are used. It is observed that bes results are obtained for combination of learning ratte of 0.05 and 15 number of epochs.

``` python
learning_rate = 0.01
batch_size = 64
num_epochs = 15
steps_per_epoch = 140
validation_steps = 30
workers = 2
```
#### Explaination about hyper parameter tuning.
<br/>

As the underlying error optimization function uses stochastic batch gradient descent funtion, the significance of learning rate and number of epochs can be similarly related to that of simple continuous regression learning. For stochastic gradient descent optimization, few thumb rules, that can be remembered are pointed below. <br/>

* Keep training network till the error fuction reaches the vicintiy of minimum error value. Low number of epochs give less number of chances to reeach miniimu error value. However on other side with high number of epochs we have risk of overfitting, as with each repetiton model is getting tied to training data and is no more a generalized model. Such overfitting cases can be identified using  validation error. <br/>

* Similarly, low training rate will take more number of epochs to reach minimum value. More epohs mean more learning time  and compute power. However, with higher learning rate we have risk of overjumping the minimum error value. <br/>
To address learning  rate tuning issue, techniques have been proposed to use variable leaning rate instead of single learning rate. Broad idea behind this approach is to use higher learrning rates initillay and as network approahes minimum value decrease the learning rate. Apart from varying learning rate with time, we also have techniques to vary the learning rate for each weight parameter. <br/>
In follow me deep learning project we have taken this approach to tune the learning rate parameter. This can be achived by <b>adagrad optimizer</b>. Even though adagrad optimizer tunes the learning rate intelligently with iterations, the initial learning rate supplied to the optimizer plays critical role in the rate at which minimum error value is obtained. <br/>

The above insights are explained with  below results from the project. <br/>

| Case        | Learning rate           | epochs  |
| ------------- |:-------------:| -----:|
| case1      | 0.01 | 5 |
| case2       | 0.005      |   5 |
| case3(solution case)  | 0.01     |    15 |
| case4  | 0.01      |    20 |

#### Case 1:
#### Case 2:
#### Case 3:
#### Case 4:


### Future work

* Collect more data related to specal cases. One case being  hero is far from the camera and is in the crowd. IOU metric is affected significantly by these cases. 
* Learn techniques and methods to undestand the intenal reresentation of network. This will help to tune the paraameters more sysematically than doing it by brute force approach.