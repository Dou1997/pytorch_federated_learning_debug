from utils.models import *
from torch.utils.data import DataLoader
from utils.fed_utils import assign_dataset, init_model
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
import matplotlib.pyplot as plt
import random
import os
from preprocessing.baselines_dataloader import load_data, ChestXrayDataset
from torchvision import transforms
import cv2
import torchvision.transforms.functional as TF


class FedClient(object):
    def __init__(self, name, epoch, dataset_id, model_name):
        """
        Initialize the client k for federated learning.
        :param name: Name of the client k
        :param epoch: Number of local training epochs in the client k
        :param dataset_id: Local dataset in the client k
        :param model_name: Local model in the client k
        """
        # Initialize the metadata in the local client
        self.target_ip = '127.0.0.3'
        self.port = 9999
        self.name = name

        # ...
        self.dataset_name = dataset_id  # store the dataset id/name in a variable
        # ...

        # Initialize the parameters in the local client
        self._epoch = epoch
        self._batch_size = 50
        self._lr = 0.001
        self._momentum = 0.9
        self.num_workers = 2
        self.loss_rec = []
        self.n_data = 0

        # Initialize the local training and testing dataset
        self.trainset = None
        self.test_data = None

        # Initialize the local model
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.param_len = sum([np.prod(p.size()) for p in model_parameters])

        # Training on GPU
        gpu = 0
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    def load_trainset(self, trainset):
        """
        Client loads the training dataset.
        :param trainset: Dataset for training.
        """
        self.trainset = trainset
        self.n_data = len(trainset)

    def update(self, model_state_dict):
        """
        Client updates the model from the server.
        :param model_state_dict: Global model.
        """
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        self.model.load_state_dict(model_state_dict)

    def train(self):
        """
        Client trains the model on local dataset
        :return: Local updated model, number of local data points, training loss
        """

        print("len of trainset is ", len(self.trainset))
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)
    
        self.model.to(self._device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        #loss_func = nn.CrossEntropyLoss()
        if self.dataset_name == 'ChestXrays':
            loss_func = torch.nn.BCEWithLogitsLoss()
        else:
            loss_func = torch.nn.CrossEntropyLoss()
        
        # Insert the snippet here to inspect the shapes:
        sample_x, sample_y = next(iter(train_loader))
        print("Sample x shape:", sample_x.shape)
        print("Sample y shape:", sample_y.shape)

        # Training process
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    # b_x = x[:, 0].to(self._device) # tring to fix the bug 50 2 1 224 224  this 5 dimension problem
                    #b_x = x.to(self._device)  # Tensor on GPU
                    #b_y = y.to(self._device)  # Tensor on GPU
                    b_x = x[:, 0].to(self._device)
                    b_y = y[:, 0].to(self._device)
                    # Add the print statements here
                    
                    #print("First b_x shape:", b_x[0].shape)
                    #print("First b_x value:\n", b_x[0])

                    #print("First b_y shape:", b_y[0].shape)
                    #print("First b_y value:\n", b_y[0])



                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.float())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()
    
    def generate_and_save_heatmap(self, client_id, decoded_labels, original_image, target, img_path=os.path.join(os.path.dirname(__file__), "../figures/heatmap")):
        #c mean cam 
        original_image_c = original_image
        target_c = target
        original_image_array = np.array(original_image_c)
        #print(f'original_image shape: {original_image_array.shape}')
        original_image_pil = TF.to_pil_image(original_image_c)
        # for Grad-CAM to set target layer
        target_layer = [self.model.layer4[-1]]
        # init GradCAM and generate heatmap
        gradcam = GradCAM(model=self.model, target_layers=target_layer, use_cuda=True)
        # make sure original_image_array formal is HWC 
        if original_image.shape[0] == 3:  # check if its  CHW (channel height weight)
            original_image_array = original_image.permute(1, 2, 0).numpy()  # 转换为 HWC 格式并转换为 NumPy 数组
        else:
            original_image_array = np.array(original_image)
        
        for class_index in range(len(decoded_labels)):  # 循环遍历所有类别的索引 for loop to search each index for all classes
            target_object = ClassifierOutputTarget(class_index)  # 为每个类别创建目标对象 setup target for each class
            input_tensor = transforms.Compose([transforms.ToTensor(),transforms.Grayscale(num_output_channels=1),])(original_image_pil).unsqueeze(0).to(self._device)
            grayscale_cam = gradcam(input_tensor, targets=[target_object])  # 生成每个类别的CAM generade cam for each class
            # Convert grayscale_cam to a uint8 mask
            mask = np.uint8(grayscale_cam * 255)
            mask = np.squeeze(mask)
            visualization = show_cam_on_image(original_image_array, mask, use_rgb=True)
            # Save the overlay visualization.
            plt.figure(figsize=(original_image_array.shape[1]/100, original_image_array.shape[0]/100), dpi=100)
            plt.imshow(visualization)
            plt.text(10, 10, decoded_labels[class_index], color='white', fontsize=12, weight='bold')
            plt.axis('off')  # Turn off the axis.
            plt.savefig(f"{img_path}_client_{client_id}_class_{class_index}.png", bbox_inches='tight', pad_inches=0)
            plt.close()
            
            
            # After you have your grayscale_cam and mask, add the metric evaluation code
            cam_metric = ROADMostRelevantFirst(percentile=75)
            scores, perturbation_visualizations = cam_metric(input_tensor, 
                                                            grayscale_cam, 
                                                            [target_object], 
                                                            self.model, 
                                                            return_visualization=True)
    
            perturbation_visualization = perturbation_visualizations[0] # Assuming one image in batch
            #print(f'perturbation_visualization before normalization is {perturbation_visualization}')
            #perturbation_visualization = perturbation_visualization.float() / 255.0
    
            
            # Now create a visualization using the perturbation_visualization
            plt.figure(figsize=(original_image_array.shape[1]/100, original_image_array.shape[0]/100), dpi=100)
            perturbation_visualization = perturbation_visualization.cpu().squeeze(0).numpy()
            plt.imshow(perturbation_visualization, cmap='jet')
            plt.title(f"Class {decoded_labels[class_index]} - Score Change: {scores[0]:.2f}")
            plt.axis('off')
            plt.savefig(f"{img_path}_client_{client_id}_perturbation_class_{class_index}.png", bbox_inches='tight', pad_inches=0)
            plt.close()
            

