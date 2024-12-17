import flwr as fl
import torch
from collections import OrderedDict
import models.flowerfed_pix2pix_model as model
from train_fed import train
from test_fed import test
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_mode

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    #probably have to add bot D and G parameters
    #this doesn't work since separate models
    #todo change the model to work with both models at once either in tuple or dict form
    def __init__(self, opt_train, opt_test):
        self.opt_train = opt_train
        self.train_data = create_dataset(opt_train)
        self.train_data = self.train_data
        self.opt_test = opt_test
        self.test_data = create_dataset(opt_test)
        #creates a new model without any parameters(initial parameters sent by server)
        self.net = create_model(opt_train)
        self.net.setup(opt_train)

    def get_parameters(self, config):
        generator, discriminator = self.net.state_dict()
        g = [val.cpu().numpy() for _, val in generator.items()]
        d = [val.cpu().numpy() for _, val in discriminator.items()]
        model_weights = g + d
        return model_weights
    
    def set_parameters(self, parameters):
        generator, discriminator = self.net.state_dict()
        len_gparam = len([val.cpu().numpy() for _, val in generator.items()])
        len_dparam = len([val.cpu().numpy() for _, val in discriminator.items()])
        params_dict = zip(generator.keys(), parameters[:len_gparam])
        gstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        params_dict = zip(discriminator.keys(), parameters[len_gparam:len_dparam+len_gparam])
        dstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # params_dict = zip(self.g_ema.state_dict().keys(), parameters[-len_emaparam:])
        # g_emastate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(gstate_dict, dstate_dict)
        # self.generator.load_state_dict(gstate_dict, strict=False)
        # self.discriminator.load_state_dict(dstate_dict, strict=False)
        # self.g_ema.load_state_dict(g_emastate_dict, strict=False)

    def fit(self, parameters, config):
        self.net.setup(self.opt_train)
        self.set_parameters(parameters)
        size = train(self.net, self.train_data, self.opt_train)
        return self.get_parameters(config={}), size, {}
    def evaluate(self, parameters, config):
        # self.net.setup(self.opt_test)
        print("eval1")
        self.set_parameters(parameters)
        print("eval2")
        # test(self.net, self.test_data, self.opt_test)
        print("eval3")
        # return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}
        return float(0), 1, {"accuracy": float(0)}

# Start Flower client
opt_train = TrainOptions().parse()
opt_test = TestOptions().parse()
# client=FlowerClient(opt)
fl.client.start_numpy_client(server_address="127.0.0.1:8080",
                             client=FlowerClient(opt_train, opt_test))
# client = ClientGAN(args, generator, discriminator, g_ema, g_optim, d_optim, train_loader)
# fl.client.start_numpy_client("127.0.0.1:8080", client)
# train(net, train_data, opt_train)