# Dual-path multi-scale context dense aggregation network for retinal vessel segmentation

## This is an implementation of the [MCDAU-Net](https://www.sciencedirect.com/science/article/abs/pii/S0010482523007345?via%3Dihub).

## Environment Configuration：
```
* Python3.9
* Pytorch1.10
* Best trained with GPUs
```

## File structure：
```
  ├── core.models: Build MCDAU-Net Model Code
  ├── core.utils: Read the dataset and calculate the mean and standard deviation
  ├── core.blocks: Build the MCDAU-Net module code
  ├── main.py: Model training
  └── test.py: Model testing and save results
```
