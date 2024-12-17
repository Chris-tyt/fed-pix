import flwr as fl
import torch
# import models.flowerfed_pix2pix_model as model
from models import create_model
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from test_fed import test

from typing import Dict, Optional, Tuple
from collections import OrderedDict

def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "rnd": rnd,
        "batch_size": 16,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config

def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"rnd": rnd, "val_steps": val_steps}

def set_parameters(net, parameters):
    generator, discriminator = net.state_dict()
    
    len_gparam = len([val.cpu().numpy() for _, val in generator.items()])
    len_dparam = len([val.cpu().numpy() for _, val in discriminator.items()])
    params_dict = zip(generator.keys(), parameters[:len_gparam])
    gstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    params_dict = zip(discriminator.keys(), parameters[len_gparam: len_dparam + len_gparam])
    dstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    # params_dict = zip(self.g_ema.state_dict().keys(), parameters[-len_emaparam:])
    # g_emastate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    net.load_state_dict(gstate_dict, dstate_dict)

def get_evaluate_fn(model, opt):
    """Return an evaluation function for server-side evaluation."""

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        # params_dict = zip(model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # model.load_state_dict(state_dict, strict=True)
        test_data = create_dataset(opt)
        set_parameters(model, parameters)
        test(model, test_data, opt, str(server_round))
        return float(1), {}

    return evaluate

opt_train = TrainOptions().parse()
opt_test = TestOptions().parse()

net = create_model(opt_train)
net.setup(opt_train)

# please modify if using a different model
generator, discriminator = net.state_dict()

#change this to load a model from a point in memory if you want to use a past model
g = [val.cpu().numpy() for _, val in generator.items()]
d = [val.cpu().numpy() for _, val in discriminator.items()]

model_weights = g + d

# fl.server.start_server(
 # server_address="0.0.0.0:8080",
 # config=fl.server.ServerConfig(num_rounds=1),
 # strategy=fl.server.strategy.FedAvg(min_available_clients=1,
 # min_evaluate_clients=1,
 # min_fit_clients=1,
 # initial_parameters=fl.common.ndarrays_to_parameters(model_weights),
 # evaluate_fn=get_evaluate_fn(net, opt_test),
 # on_fit_config_fn=fit_config,
 # on_evaluate_config_fn=evaluate_config,
 # ),
 # )
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=fl.server.strategy.FedAvg(
        initial_parameters=fl.common.ndarrays_to_parameters(model_weights),
        evaluate_fn=get_evaluate_fn(net, opt_test),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    ),
 )
