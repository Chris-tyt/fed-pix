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
from options.test_options import TestOptions


from data import create_dataset
from models import create_model
from train_fed import train 

# Define Flower client for Federated Learning
class FederatedPix2PixClient(fl.client.NumPyClient):
    # def __init__(self, opt_train, opt_test):
    def __init__(self, opt_train, opt_test):
        # Initialize training and testing options
        self.opt_train = opt_train
        self.opt_test = opt_test
        
        # Create training and testing datasets
        self.train_data = create_dataset(opt_train)
        self.test_data = create_dataset(opt_test)
        
        dataset_size = len(self.train_data)
        test_dataset_size = len(self.test_data)
        print('The number of training images = %d' % dataset_size)
        print('The number of testing images = %d' % test_dataset_size)
        
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

        # 使用服务器配置的参数
        # batch_size = config.get("batch_size", self.opt_train.batch_size)
        batch_size = config.get("batch_size", self.opt_train.batch_size)
        local_epochs = config.get("local_epochs", 1)
        
        # Training loop
        total_iters = 0
        print(f"Training with batch_size: {batch_size}, local_epochs: {local_epochs}")
        print(f"Dataset size: {len(self.train_data)}")
        
        for epoch in range(local_epochs):
            self.net.update_learning_rate()
            for i, data in enumerate(self.train_data):
                print(f'epoch: {epoch}, data num: {i}')
                total_iters += batch_size
                self.net.set_input(data)
                self.net.optimize_parameters()
        
        return self.get_parameters(config), len(self.train_data), {}

    def evaluate(self, parameters, config):
        print("Starting client evaluation...")
        # Load the parameters from the server
        self.net.setup(self.opt_test)  # 使用测试配置重新设置网络
        self.set_parameters(parameters, config)
        
        # Set evaluation mode
        self.net.eval()
        
        # Initialize metrics
        total_images = 0
        metrics = {}
        
        with torch.no_grad():
            for i, data in enumerate(self.test_data):
                if i >= self.opt_test.num_test:
                    break
                
                self.net.set_input(data)
                self.net.test()
                
                # Get current losses
                losses = self.net.get_current_losses()
                for name, loss in losses.items():
                    metrics[name] = metrics.get(name, 0) + float(loss)
                
                total_images += 1
        
        # Average the metrics
        if total_images > 0:
            for key in metrics:
                metrics[key] /= total_images
        
        # Return format: (loss, num_samples, metrics_dict)
        return float(metrics.get('G_L1', 0.0)), total_images, metrics
        

# if __name__ == '__main__':
opt_train = TrainOptions().parse()
opt_test = TestOptions().parse()
fl.client.start_client(server_address="127.0.0.1:8080", 
                        client=FederatedPix2PixClient(opt_train, opt_test).to_client())
