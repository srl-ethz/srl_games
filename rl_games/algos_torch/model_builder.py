from rl_games.common import object_factory
import rl_games.algos_torch
from rl_games.algos_torch import network_builder
from rl_games.algos_torch import models

NETWORK_REGISTRY = {}
MODEL_REGISTRY = {}

# these functions are just needed to maintain compatibility with IsaacGymEnvs
def register_network(name, target_class):
    pass

def register_model(name, target_class):
    pass

class NetworkBuilder:
    def __init__(self):
        self.network_factory = object_factory.ObjectFactory()
        self.network_factory.set_builders(NETWORK_REGISTRY)
        self.network_factory.register_builder('actor_critic', lambda **kwargs: network_builder.A2CBuilder())

    def load(self, params):
        network_name = params['name']
        network = self.network_factory.create(network_name)
        network.load(params)

        return network


class ModelBuilder:
    def __init__(self):
        self.model_factory = object_factory.ObjectFactory()
        self.model_factory.set_builders(MODEL_REGISTRY)
        self.model_factory.register_builder('continuous_a2c_logstd',
                                            lambda network, **kwargs: models.ModelA2CContinuousLogStd(network))
        self.network_builder = NetworkBuilder()

    def load(self, params):
        model_name = params['model']['name']
        network = self.network_builder.load(params['network'])
        model = self.model_factory.create(model_name, network=network)
        return model
