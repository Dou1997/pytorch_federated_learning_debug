# pytorch_federated_learning_debug

## Federated learning algorithms
Original GitHub repositories to reference:
- [pytorch_federated_learning](https://github.com/rruisong/pytorch_federated_learning)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- [FL_Attention](https://github.com/LeiserKIT/FL_Attention)

## Introduction about Federated Learning Task

### Operation Steps:
1. On the master branch, you will find the code.
2. here:https://www.kaggle.com/datasets/ashery/chexpert/data  download the datasets
3. When you connect to Bwunicluster with ssh, you will encounter the login interface.
4. Use the command `salloc -p gpu_4 -n1 --gres=gpu:1 -t 120` for GPU usage (120 minutes), alternatively, `gpu_8` can also be used.
5. Load the modules with `module load devel/cuda/11.8` and `module load compiler/gnu/12.1` to get CUDA and GNU.
6. Navigate to the `pytorch_federated_learning` workspace and execute `python fl_main.py --config "./config/test_config.yaml"` to start training the Neural Networks on the ChestXray dataset.
7. Further details can be found in the `test_config.yaml` file. Ensure the dataset folder is in the same directory as `pytorch_federated_learning`.
8. After running the python command, the heatmap of the client part neural networks will be stored in the `figures` folder within `pytorch_federated_learning`.
9. After training the neural networks, execute `python postprocessing/eval_main.py -rr 'results'` to store the performance results in the `figures` folder.

### Attention
#### Issues Not Solved
1. Integrating heatmap with server neural network part.
2. Upon executing `python postprocessing/eval_main.py -rr 'results'`, the performance results do not match the configurations in `test_config.yaml`. Debugging of `recoder.py` is an ongoing task indicating remaining bugs.

### Advice
1. For any dataset issues with ChestXray, inspect `baselines_dataloader.py`, focusing on `ChestXrayDataset` class and `load_data` function.
2. For heatmap issues, refer to `client_base.py` and review the `generate_and_save_heatmap` function. The usage of this function is documented in `fl_main.py`.
3. Ensure that the dimensions of the neural network's output layer match the dimensions of your labels.

## Current Circumstances
- Utilizing ResNet50 for multi-object image classification on the CheXpert medical imaging dataset from Stanford University. Note that the input images are grayscale, not RGB, and pairs of images are retrieved from the trainset simultaneously.
- Testing the accuracy of the FedAvg and FedProx aggregation algorithms, applied to the CheXpert dataset.
- Enhancing the aggregation accuracy and interpretability of the federated learning algorithm by integrating the heatmap with FedAvg using Pytorch-Grad-CAM.
