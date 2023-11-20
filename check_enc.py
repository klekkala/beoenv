
from ray.rllib.models import ModelCatalog
import envs
import torch
from envs import SingleAtariEnv
from atari_vae import Encoder, TEncoder
from ray.rllib.policy.policy import Policy
from models.beogymmodels import SingleBeogymModel, BeogymCNNV2PlusRNNModel, FrozenBackboneModel, SingleImageModel, ComplexNet
from models.atarimodels import SingleAtariModel
import torch
import sys

model_path = sys.argv[1]
torch_path = model_path[model_path.find('CHAN')-1:model_path.find('pt')+2]
ModelCatalog.register_custom_model("model", SingleAtariModel)
ModelCatalog.register_custom_model("Single", SingleImageModel)
ModelCatalog.register_custom_model("FrozenBackboneModel", FrozenBackboneModel)
ModelCatalog.register_custom_model("ComplexNet", ComplexNet)
model_path = f"/lab/kiran/logs/rllib/beogym/notemp/{model_path}/checkpoint/"

model = Policy.from_checkpoint(model_path)

weights = model.get_weights()
# encoder_weight = weights['encoder.encoder.1.weight']
encoder_bias = weights['encoder.encoder.1.bias']
print(encoder_bias)


state_dict = torch.load('/lab/kiran/ckpts/pretrained/beogym/'+torch_path)

print(state_dict['model_state_dict']['encoder.1.bias'])