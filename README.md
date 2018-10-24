# discrete-matrix-factorization
This repository contains the source code for the paper entitled "Learning Discrete Matrix Factorization Models"

## Prerequisites
```
- Tensorflow 
- Numpy 
```
The code has been tested in Ubuntu 14.04 and MacOSX, with
```
- Tensorflow v.1.2.1
- Numpy v.1.12.1 
```
You should be able to run the code with newer versions of Tensorflow and Numpy as well. 

## Usage
### Prepare data
A sample split (75% training, 25% testing) of the MovieLens1M dataset is included (inside folder ./data/MovieLens1M/):
- rating.npz: original matrix 
- train_mask.npz: training mask, same shape as the original matrix
- val_mask.npz: validation mask, same shape as the original matrix
- test_mask.npz: testing mask, same shape as the original matrix
all these matrices are saved as Numpy sparse csr_matrix. 

Alternatively, you can write your own DataLoader to load and feed the model during training, with data prepared in another way.

### Configurations
The configs/configs_dmfd.py file sets all the default configurations and hyper-parameters.
There are some other hyperparameters inside train.py and test.py
To use your own configurations, you can either edit these files, or override the default configurations using flags.

For example:
```
python train.py --n_epochs=1000 --initial_lr=0.01
```

### Training
After preparing your dataset and set all the necessary hyper-parameters, you are ready to train your model.
Run train.py, together with our hyper-parameters, to start the training. For example:
```
python train.py --data_dir=./data/MovieLens1M/ --output_basedir=./outputs/
```
The trained model will be saved into outputs/snapshots.
To save the training log, which can be inspected using tensorboard, you need to set:
```
    write_summary = True
```
in the configuration file.

### Testing
After training the model, you can test it by calling test.py, with the default parameters or with your own parameters. For example:
```
python test.py --data_dir=./data/MovieLens1M/ --snapshot_dir=./outputs/snapshots/
```
## Related
If you just want to produce continuous output, you may want to check our paper: "Extendable Neural Matrix Completion" (published at ICASSP 2018)
The source code is available at: https://github.com/nmduc/neural-matrix-completion

## License
This project is licensed under the MIT License - see the LICENSE.md file for details

## Reference
If you find the source code useful, please cite us:
```
D. M. Nguyen, E. Tsiligianni and N. Deligiannis, "Learning Discrete Matrix Factorization Models," in IEEE Signal Processing Letters, vol. 25, no. 5, pp. 720-724, May 2018.
```
