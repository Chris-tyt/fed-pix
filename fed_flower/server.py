import flwr as fl
import torch
import warnings
import sys
import os

from collections import OrderedDict

# 忽略 DeprecationWarning 类型的警告
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# 将上一级文件夹添加到系统路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from options.train_options import TrainOptions
from data import create_dataset
from models import create_model


def fit_config(rnd: int):
    """Return training configuration dict for each round."""
    config = {
        "rnd": rnd,
        "batch_size": 16,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config

def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round."""
    val_steps = 5 if rnd < 4 else 10
    return {"rnd": rnd, "val_steps": val_steps}

def set_parameters(net, parameters):
    # Set parameters for generator and discriminator
    generator, discriminator = net.my_state_dict()
    len_gparam = len([val.cpu().numpy() for _, val in generator.items()])
    len_dparam = len([val.cpu().numpy() for _, val in discriminator.items()])
    
    # Set generator parameters
    params_dict = zip(generator.keys(), parameters[:len_gparam])
    gstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    
    # Set discriminator parameters
    params_dict = zip(discriminator.keys(), parameters[len_gparam: len_gparam + len_dparam])
    dstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    net.my_load_state_dict(gstate_dict, dstate_dict)

def get_evaluate_fn(model, opt):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round: int, parameters, config):
        # Update model with the latest parameters
        set_parameters(model, parameters)
        
        # Evaluation loop
        test_data = create_dataset(opt)
        # total = len(test_data)
        # correct = 0
        # with torch.no_grad():
        #     for data in test_data:
        #         model.set_input(data)
        #         model.forward()
        #         # Placeholder: Add evaluation logic here (e.g., accuracy calculation)
        #         correct += 1  # Replace with actual metric calculation
        # # return float(correct) / total, {}
        return float(1), {}
    return evaluate

# Parse options
opt_train = TrainOptions().parse()
# opt_test = TrainOptions().parse()  # For simplicity, using the same options for test; customize if needed

# Create model
net = create_model(opt_train)
net.setup(opt_train)

# Get initial model weights
generator, discriminator = net.my_state_dict()
# generator= net.my_state_dict()
g = [val.cpu().numpy() for _, val in generator.items()]
d = [val.cpu().numpy() for _, val in discriminator.items()]
print(len(g),'--',len(d))
# print(g[1])
# print(d[1])
# model_weights = g
model_weights = g + d

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=fl.server.strategy.FedAvg(
        initial_parameters=fl.common.ndarrays_to_parameters(model_weights),
        # evaluate_fn=get_evaluate_fn(net, opt_test),
        # on_fit_config_fn=fit_config,
        # on_evaluate_config_fn=evaluate_config,
    ),
)
