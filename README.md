# Janus Extension #

This repo contains a Keras implementation of the paper,     
[VGGFace2: A dataset for recognising faces across pose and age (Cao et al., FG 2018)](https://arxiv.org/abs/1710.08092).

### Dependencies
- [Python 2.7.15](https://www.continuum.io/downloads)
- [Keras 2.2.4](https://keras.io/)
- [Tensorflow 1.8.0](https://www.tensorflow.org/)

### Data
The dataset used for the experiments are

- [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) [1]

### Testing the model
To test a specific model on the IJB dataset, 
for example, the model trained with ResNet50 trained by sgd with softmax, and feature dimension 512

- python src/predict.py --net resnet50 --batch_size 64 --gpu 2 --loss softmax --aggregation avg --resume ../model/resnet50_softmax_dim512/weights.h5 --feature_dim 512

### Citation
```
@InProceedings{Cao18,
  author       = "Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman ",
  title        = "VGGFace2: A dataset for recognising face across pose and age",
  booktitle    = "International Conference on Automatic Face and Gesture Recognition, 2018.",
  year         = "2018",
}
```

