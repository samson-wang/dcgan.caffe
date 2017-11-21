# dcgan.caffe: A pure caffe-python implementation of [DC-GAN](https://github.com/soumith/dcgan.torch)

As far as I know, there is no light-weight implementation of DCGAN based on caffe.

Inspired by [DeePSiM](http://lmb.informatik.uni-freiburg.de/resources/binaries/arxiv2016_alexnet_inversion_with_gans/release_deepsim_v0.5.zip) implementation, a few lines of python code can train the dcgan model quickly without any hack in caffe core lib ([Dosovitskiy](https://github.com/dosovits/caffe-fr-chairs/tree/deepsim) has already done this. However, I think the code could be merged back to master branch).

## Dependency
You will need to compile the [deepsim-caffe-branch](https://github.com/dosovits/caffe-fr-chairs/tree/deepsim). And make sure your `PYTHONPATH` point to it.

The deepsim-caffe only support cudnn-4.0. If disable the cudnn engine and replace some convolution layers with the master branch, a latest cudnn and cuda will work fine.

## Training
For face generator, please prepare [celebA](https://github.com/soumith/dcgan.torch#11-train-a-face-generator-using-the-celeb-a-dataset) dataset as the link said. Then make a train list file and put it in the data.prototxt.

Just typing
```
python train.py
```

## Train file list
Each line has two columns seperated by space. The second column indicates the label of the corresponding image. Actually the label could be all zeros, since the training only need the images themselves to be the targets. 

The file should look like

```
/data/Repo/dcgan.torch/celebA/img_align_celeba/000001.jpg 0
/data/Repo/dcgan.torch/celebA/img_align_celeba/000002.jpg 0
/data/Repo/dcgan.torch/celebA/img_align_celeba/000003.jpg 0
/data/Repo/dcgan.torch/celebA/img_align_celeba/000004.jpg 0
/data/Repo/dcgan.torch/celebA/img_align_celeba/000005.jpg 0
/data/Repo/dcgan.torch/celebA/img_align_celeba/000006.jpg 0
/data/Repo/dcgan.torch/celebA/img_align_celeba/000007.jpg 0
/data/Repo/dcgan.torch/celebA/img_align_celeba/000008.jpg 0
/data/Repo/dcgan.torch/celebA/img_align_celeba/000009.jpg 0
/data/Repo/dcgan.torch/celebA/img_align_celeba/000010.jpg 0
/data/Repo/dcgan.torch/celebA/img_align_celeba/000011.jpg 0
/data/Repo/dcgan.torch/celebA/img_align_celeba/000012.jpg 0
/data/Repo/dcgan.torch/celebA/img_align_celeba/000013.jpg 0
...
```
## Trouble shooting
- All your images' size should be `64x64`, which is specified in https://github.com/samson-wang/dcgan.caffe/blob/master/discriminator.prototxt

- Please use the *deepsim* branch

## Visualization
To view the model result by
```
python generate.py generator.prototxt snapshots_test/4000/generator.caffemodel
```

The visualizations of the models at iteration 3000 and 4000 are as following:

![3000](output/iter_3000.png)

![4000](output/iter_4000.png)
