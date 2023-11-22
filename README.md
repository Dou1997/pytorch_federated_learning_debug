# pytorch_federated_learning_debug
Federated learning algorithms
original Github website which need to look at
1:https://github.com/rruisong/pytorch_federated_learning
2:https://github.com/jacobgil/pytorch-grad-cam
3:https://github.com/LeiserKIT/FL_Attention
---------------------------------------------
Introduction about Federated Learning Task
---------------------------------------------
Operation following :
1.into the branch master you will find out the code 
2.once you connect Bwunicluster with ssh you will stuck in the Login interface
3. usding command :salloc -p gpu_4 -n1 --gres=gpu:1 -t 120  (this is for the GPU using assignment, 120 means 120 mins, we can also choose gpu_8 to use.
4. module load devel/cuda/11.8 and module load compiler/gnu/12.1 will provide you cuda and gnu
5. cd pytorch_federated_learning workspace and run python fl_main.py --config "./config/test_config.yaml" then you will start train the Neural Networks which base on ChestXray dataset.
6. more details you can see the test_config.yaml file. and The dataset folder should be in the same directory as pytorch_federated_learning.
7. once you run the python command, the heatmap of the client part neural networks will store in the figures folder which inside pytorch_federated_learning.
8. once you trained the neural nerworks, run python postprocessing/eval_main.py -rr 'results',then the performance results of the neural network will store in the figures folder.
--------------------------------------------
Attention
NOT Slolving Problem 
1. Combine heatmap with server neural network part
2. once you run python postprocessing/eval_main.py -rr 'results', the results of performance neural networks is not matched with the set in the test_config.yaml and I have been starting to debug the recoder.py and it’s still a ongoing task which means there are still bugs exist.
--------------------------------------------
Advice 
1.If you have any issues regarding the dataset chestxray, you can dig into baselines_dataloader.py and look at class ChestXrayDataset and function load_data.
2.If you have any issues regarding the heatmap, you can dig into client_base.py and look at function generate_and_save_heatmap and the way how to recall this function is in fl_main.py
3.you need to make sure the dimensions of the output layer is match with the dimensions of your labels.
--------------------------------------------
Current circumstances 
• Using ResNet50 to process multi-objects Image classification on CheXpert medical imaging Dataset from Stanford
University.(the input is gray image not RGB image and each time when you get the image from trainset to train neural networks, you will get two images at same time.)
• Tesing the accuracy of the aggregation algorithm of FedAvg and FedProx two federated learning algorithms based on the
CheXpert Datsets.
• Combining heatmap with FedAvg by using Pytorch-Grad-CAM to improve the aggregation accuracy of the federated
learning algorithm by advancing the interpretability of AI Models.
--------------------------------------------
