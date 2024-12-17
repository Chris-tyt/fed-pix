import flwr as fl
import torch
import sys
import os
import time
import warnings

from collections import OrderedDict

# 忽略 DeprecationWarning 类型的警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 将上一级文件夹添加到系统路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from options.train_options import TrainOptions
# from options.test_options import TestOptions


from data import create_dataset
from models import create_model
from train_fed import train 

# Define Flower client for Federated Learning
class FederatedPix2PixClient(fl.client.NumPyClient):
    # def __init__(self, opt_train, opt_test):
    def __init__(self, opt_train):
        # Initialize training and testing options
        self.opt_train = opt_train
        # self.opt_test = opt_test
        
        # Create training and testing datasets
        self.train_data = create_dataset(opt_train)
        dataset_size = len(self.train_data)    # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)
        # self.test_data = create_dataset(opt_test)
        
        # Create model
        self.net = create_model(opt_train)
        self.net.setup(opt_train)
    
    def get_parameters(self, config):
        # Get the parameters from the generator and discriminator
        generator, discriminator = self.net.my_state_dict()
        g = [val.cpu().numpy() for _, val in generator.items()]
        d = [val.cpu().numpy() for _, val in discriminator.items()]
        model_weights = g + d
        return model_weights

    def set_parameters(self, parameters, config):
        # 设置生成器和判别器的参数
        generator, discriminator = self.net.my_state_dict()
        len_gparam = len([val.cpu().numpy() for _, val in generator.items()])
        len_dparam = len([val.cpu().numpy() for _, val in discriminator.items()])
        print(len_gparam, len_dparam)
        
        # 设置生成器参数
        params_dict = zip(generator.keys(), parameters[:len_gparam])
        gstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        # 检查生成器参数的形状
        for k, v in gstate_dict.items():
            if v.shape != generator[k].shape:
                print(f"生成器参数 {k} 的形状不匹配：预期 {generator[k].shape}，实际 {v.shape}")
                return  # 或者抛出错误

        # 设置判别器参数
        params_dict = zip(discriminator.keys(), parameters[len_gparam: len_gparam + len_dparam])
        dstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        # 检查判别器参数的形状
        for k, v in dstate_dict.items():
            if v.shape != discriminator[k].shape:
                print(f"判别器参数 {k} 的形状不匹配：预期 {discriminator[k].shape}，实际 {v.shape}")
                return  # 或者抛出错误

        self.net.my_load_state_dict(gstate_dict, dstate_dict)

    def fit(self, parameters, config):
        # print(parameters)
        # Load the parameters from the server
        print('--------------1---------------')
        self.net.setup(self.opt_train)
        self.set_parameters(parameters, config)

        # Training loop
        total_iters = 0
        # dataset = create_dataset(self.opt_train)
        print(len(self.train_data))
        for epoch in range(self.opt_train.epoch_count, self.opt_train.n_epochs + self.opt_train.n_epochs_decay + 1):
            self.net.update_learning_rate()
            for i, data in enumerate(self.train_data):
                print(f'epoch: {epoch}, data num: {i}')
                total_iters += self.opt_train.batch_size
                self.net.set_input(data)
                self.net.optimize_parameters()
        return self.get_parameters(config), len(self.train_data), {}

    def evaluate(self, parameters, config):
        print("in client eval")
        # # Load the parameters from the server
        # self.set_parameters(parameters)
        
        # # Evaluation loop (using test data)
        # total = len(self.test_data)
        # correct = 0
        # with torch.no_grad():
        #     for data in self.test_data:
        #         self.net.set_input(data)
        #         self.net.forward()
        #         # Placeholder: Here you can add any metric evaluation logic based on your specific needs
        #         # For example, calculating accuracy or L1 loss
        #         correct += 1  # Replace with actual metric calculation
        # return float(correct) / total, total, {}
        return float(0), 1, {"accuracy": float(0)}
        

# if __name__ == '__main__':
opt_train = TrainOptions().parse()
# opt_test = TestOptions().parse()
fl.client.start_client(server_address="127.0.0.1:8080", 
                        client=FederatedPix2PixClient(opt_train).to_client())
