from fed_baselines.client_base import FedClient
import copy
from utils.models import *

from torch.utils.data import DataLoader


class FedProxClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name):
        super().__init__(name, epoch, dataset_id, model_name)
        self.mu = 0.1

    def train(self):
        """
        Client trains the model on local dataset using FedProx
        :return: Local updated model, number of local data points, training loss
        """
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

        self.model.to(self._device)
        global_weights = copy.deepcopy(list(self.model.parameters()))

        '''
        # Debugging information for train_loader:
        print(f"Length of train_loader: {len(train_loader)}")
        print(f"Length of train_loader dataset: {len(train_loader.dataset)}")
        print(f"train_loader: {train_loader}")
        print(f"dir train_loader:{dir(train_loader)}")  # 打印加载器的所有属性和方法
        '''

        '''
        try:
            for step, (b_x, b_y) in enumerate(train_loader):
                b_x = b_x.view(-1, 1, 224, 224)  # Reshape to [100, 1, 224, 224]
                b_y = b_y.view(-1, 5)  # Reshape to [100, 5]

                print(b_x.shape)
                print(f"Shape of b_x: {b_x.shape}")
                print(f"Shape of b_y: {b_y.shape}")
            # ... 其他代码
        except Exception as e:
            print(f"Error: {e}")
            print("Failed to get data from train_loader")
        '''
        '''
        try:
            for step, (x, y) in enumerate(train_loader):
                x = x.view(-1, 1, 224, 224) # Reshape to [100, 1, 224, 224]
                y = y.view(-1, 5) # Reshape to [100, 5]
                print(f"Shape of x for train_loader at step {step}: {x.shape}")
                print(f"Shape of y for train_loader at step {step}: {y.shape}")
            # ... 其他代码 ...
        except Exception as e:
            print(f"Error in train_loader: {e}")
        '''

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        loss_func = nn.CrossEntropyLoss()
        if self.dataset_name == 'ChestXrays':
            loss_func = torch.nn.BCEWithLogitsLoss()
        else:
            loss_func = torch.nn.CrossEntropyLoss()

        epoch_loss_collector = []

        # pbar = tqdm(range(self._epoch))
        for epoch in range(self._epoch):
            
            for step, (x, y) in enumerate(train_loader):
                x = x.view(-1, 1, 224, 224) # Reshape to [100, 1, 224, 224]
                y = y.view(-1, 5) # Reshape to [100, 5]
                with torch.no_grad():
                    
                    b_x = x.to(self._device)  # Tensor on GPU
                    b_y = y.to(self._device)  # Tensor on GPU
                    #print(f"b_y:{b_y}")

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y)
                    optimizer.zero_grad()

                    #fedprox
                    prox_term = 0.0
                    for p_i, param in enumerate(self.model.parameters()):
                        prox_term += (self.mu / 2) * torch.norm((param - global_weights[p_i])) ** 2
                    loss += prox_term
                    epoch_loss_collector.append(loss.item())

                    loss.backward()
                    optimizer.step()



        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()