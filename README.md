# Dual-path multi-scale context dense aggregation network for retinal vessel segmentation

## This is an implementation of the [MCDAU-Net](https://www.sciencedirect.com/science/article/abs/pii/S0010482523007345?via%3Dihub).Due to memory limitations on GitHub, each person needs to perform the chunking operation themselves. For detailed instructions on the chunking process, please refer to the paper.

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
  ├── image2patch: Split the image into patches and perform preprocessing operations
  ├── patch2iamge.py: Merge patch into image and perform test operation
  └── test.py: Model testing and save results
```
