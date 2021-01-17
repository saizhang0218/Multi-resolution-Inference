# Multi-resolution-Inference

This repository is the official implementation of "Field-Configurable Multi-resolution Inference: Rethinking Quantization" published in ASPLOS 2021. 

### Requirements

Pytorch 1.0.0 with CUDA

## Training
This repo includes the multi-reseolution training of resnet-18 and resnet-50. 

To train the meta multi-resolution model(s) in the paper, run this command:

```train
python main.py -a model_name save_path
```

> 📋model_name can be one of the following three options: resnet-18, resnet-34, resnet-50.

> 📋save_path is the location of the result folder.


## Code Explanation

> 📋The "model/resnet.py" contains the definitions of the multi-resolution resnet models.

> 📋The "model/quant_layer.py" contains the definitions of term quantization functions.

> 📋The "kernel" folder contains the definitions of the cuda kernels for term quantization.

> 📋The test accuracy will be saved in the 'accuracy' folder.
