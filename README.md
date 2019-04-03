# VGGFace2 Extension #

This repo contains a Keras implementation of the paper,     

[VGGFace2: A dataset for recognising faces across pose and age (Cao et al., FG 2018)](https://arxiv.org/abs/1710.08092).

### Dependencies
- [Python 2.7.15](https://www.continuum.io/downloads)
- [Keras 2.2.4](https://keras.io/)
- [Tensorflow 1.8.0](https://www.tensorflow.org/)

### Data
The dataset used for the experiments are

- [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) [1]

### Model

Keras Model (https://drive.google.com/file/d/1AHVpuB24lKAqNyRRjhX7ABlEor6ByZlS/view?usp=sharing),

### Note:
This model is trained with a slightly different tight crops, but I have also tested on the tight crops (as we did in the paper), and am able to get similar results (on IJBB).

TAR @ FAR = 1e-5 : 0.64 

TAR @ FAR = 1e-4 : 0.78 

TAR @ FAR = 1e-3 : 0.88 

TAR @ FAR = 1e-2 : 0.94 

TAR @ FAR = 1e-1 : 0.98

### Testing the model
To test a specific model on the IJB dataset, 
for example, the model trained with ResNet50 trained by sgd with softmax, and feature dimension 512

- python predict.py --net resnet50 --batch_size 64 --gpu 2 --loss softmax --aggregation avg --resume ../model/resnet50_softmax_dim512/weights.h5 --feature_dim 512

### Citation
```
@InProceedings{Cao18,
  author       = "Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman ",
  title        = "VGGFace2: A dataset for recognising face across pose and age",
  booktitle    = "International Conference on Automatic Face and Gesture Recognition, 2018.",
  year         = "2018",
}
```

