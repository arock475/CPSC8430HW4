In this homework assignment we were tasked with training DCGAN and WGAN models from scratch on the CIFAR 10 dataset. Before I get to the results I will briefly explain what DCGAN and WGAN are and how they function. 

DCGAN or Deep Convolutional Generative Adversarial Networks are models which are trained to generate images. The way that images are generated is by training two neural networks which compete with each other resulting in new images created from a sample training dataset. These two neural networks are typically called the Generator and the Discriminator. The generator is tasked with creating new images from an input of “random” noise while the Discriminator is tasked with determining if a given image is from the training dataset or not. This results in the generator attempting to create better images to trick the discriminator while the discriminator tries to become better at detecting the fake images. What DCGAN changes about this is several key architectural components. Generally DCGAN adds CNN architecture into a GAN model. First there is the addition of convolutional layers as opposed to pooling layers, these layers are the key in CNN architecture and function by creating a feature map from data which is typically scaled down from the true image. Next DCGANs utilize batchnorm in both the generator and the discriminator. This allows for more nuanced normalization as opposed to normalizing an entire layer at once. Next DCGANs use ReLU within the generator and leaky ReLU within the discriminator. These layers attempt to alleviate the vanishing gradient problem as well as the dying gradient problem (in the case of Leaky ReLU). Overall these changes allow for DCGAN models to behave more similarly to CNNs and attempt to increase image quality as well as to make training more stable (as GAN training is traditionally unstable and finicky).

WGAN or Wasserstein Generative Adversarial Network is another implementation of GAN models. These models are different, more so in the sense of what is being minimized. Traditionally probability distances are what is minimized in GAN models, however WGAN models minimize something called Earth Mover distances (EM distances). These EM distances functionally mean that the minimum amount of movement is used to transfer from one state to another. Why this is important is that optimizing these EM distances allows for the training of the WGAN model to be more stable. Typically GAN model training is very volatile and any small difference can cause major consequences in the training (a fact I learned during the completion of this project all too well). WGAN models alleviate some of the volatility associated with GAN models by training the EM values (but not all of it). 


To run the code utilize the following commands

```
python DCGAN_Spears.py
```
or 
```
python WGAN_Spears.py
```

You may need to use python3 depending on your environment configuration.

All input data is downloaded automaticaly by the given code
