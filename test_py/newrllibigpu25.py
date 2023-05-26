import gymnasium as gym
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from torchvision.models import resnet18, ResNet18_Weights
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from typing import Mapping, Any
import cv2
import numpy as np
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from pprint import pprint
from ray.rllib.policy.sample_batch import SampleBatch
#env = gym.make("ALE/BeamRider-v5")
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


from typing import Mapping, Any

from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule

from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT, STATE_IN
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.nested_dict import NestedDict

torch, nn = try_import_torch()



class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """
    def __init__(self, channel_in, channel_out, scale=2):
        super(ResDown, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale, scale)
        
    def forward(self, x):
        skip = self.conv3(self.AvePool(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """
    def __init__(self, channel_in, channel_out, scale=2):
        super(ResUp, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        
        self.UpNN = nn.Upsample(scale_factor = scale,mode = "nearest")
        
    def forward(self, x):
        skip = self.conv3(self.UpNN(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x
    

class Encoder(nn.Module):
    """
    Encoder block
    Built for a 3x64x64 image and will result in a latent vector of size z x 1 x 1
    As the network is fully convolutional it will work for images LARGER than 64
    For images sized 64 * n where n is a power of 2, (1, 2, 4, 8 etc) the latent feature map size will be z x n x n

    When in .eval() the Encoder will not sample from the distribution and will instead output mu as the encoding vector
    and log_var will be None
    """
    def __init__(self, channels, ch=64, z=512):
        super(Encoder, self).__init__()
        self.conv1 = ResDown(channels, ch)  # 64
        self.conv2 = ResDown(ch, 2*ch)  # 32
        self.conv3 = ResDown(2*ch, 4*ch)  # 16
        self.conv4 = ResDown(4*ch, 8*ch)  # 8
        self.conv5 = ResDown(8*ch, 8*ch)  # 4
        self.conv_mu = nn.Conv2d(8*ch, z, 2, 2)  # 2
        self.conv_log_var = nn.Conv2d(8*ch, z, 2, 2)  # 2

    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):
        x = x.float()
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #x = x.flatten(start_dim=1)

        if self.training:
            mu = self.conv_mu(x)
            log_var = self.conv_log_var(x)
            x = self.sample(mu, log_var)
        else:
            mu = self.conv_mu(x)
            x = mu
            log_var = None
        
        return x, mu, log_var

class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels, ch=64, z=512):
        super(Decoder, self).__init__()
        self.conv1 = ResUp(z, ch*8)
        self.conv2 = ResUp(ch*8, ch*8)
        self.conv3 = ResUp(ch*8, ch*4)
        self.conv4 = ResUp(ch*4, ch*2)
        self.conv5 = ResUp(ch*2, ch)
        self.conv6 = ResUp(ch, ch//2)
        self.conv7 = nn.Conv2d(ch//2, channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x 



class VAE(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """
    def __init__(self, channel_in, ch=64, z=512):
        super(VAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """
        
        self.encoder = Encoder(channel_in, ch=ch, z=z)
        self.decoder = Decoder(channel_in, ch=ch, z=z)

    def forward(self, x):
        mu = self.encoder(x)[1]
        
        #encoding, mu, log_var = self.encoder(x)
        #recon = self.decoder(encoding)
        #HARDCODED
        #return recon, mu, log_var
        #return mu
        return torch.flatten(mu, start_dim=1)


class CusEnv(gym.Env):
    def __init__(self,env_config):
        self.env = gym.make("ALE/BeamRider-v5", full_action_space=True)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8) #self.env.observation_space

    def reset(self, seed=None, options=None):
        obs,info = self.env.reset()
        return cv2.resize(obs, (84, 84)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = cv2.resize(obs, (84, 84))
        return obs, reward, terminated, truncated, info



class PPOTorchRLModule(PPORLModule, TorchRLModule):
    framework: str = "torch"

    def __init__(self, *args, **kwargs):
        TorchRLModule.__init__(self, *args, **kwargs)
        PPORLModule.__init__(self, *args, **kwargs)
        #self.blahencoder = Encoder(channels=3).cuda()
        self.blahencoder = VAE(channel_in=3, ch=64)
        #checkpoint = torch.load("/lab/kiran/ckpts/pretrained/atari/" + "STL10_ATTARI_64.pt")
        #self.blahencoder.load_state_dict(checkpoint['model_state_dict'])
        #self._weights = ResNet18_Weights.IMAGENET1K_V1
        #self.blahencoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.blahencoder.eval()
        for param in self.blahencoder.parameters():
            param.requires_grad = False
        #self._preprocess = self._weights.transforms()
        #print(self.encoder)

    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}

        # TODO (Artur): Remove this once Policy supports RNN
        if self.encoder.config.shared:
            batch[STATE_IN] = None
        else:
            batch[STATE_IN] = {
                ACTOR: None,
                CRITIC: None,
            }
        batch[SampleBatch.SEQ_LENS] = None
        
        with torch.no_grad():
            batch['obs'] = self.blahencoder(batch['obs'].cuda()).detach()
        
        #batch['obs'] = self.blahencoder(batch['obs'].cuda())
        encoder_outs = self.encoder(batchself.vf(encoder_outs[ENCODER_OUT][CRITIC]))
        # TODO (Artur): Un-uncomment once Policy supports RNN
        # output[STATE_OUT] = encoder_outs[STATE_OUT]

        # Actions
        action_logits = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits

        return output

    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        """PPO forward pass during exploration.
        Besides the action distribution, this method also returns the parameters of the
        policy distribution to be used for computing KL divergence between the old
        policy and the new policy during training.
         """
        with torch.no_grad():
            batch['obs'] = self.blahencoder(batch['obs'].cuda()).detach()
            return self._common_forward(batch)

    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        with torch.no_grad():
            batch['obs'] = self.blahencoder(batch['obs'].cuda()).detach()
        return self._common_forward(batch)

    def _common_forward(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}

        # TODO (Artur): Remove this once Policy supports RNN
        if self.encoder.config.shared:
            batch[STATE_IN] = None
        else:
            batch[STATE_IN] = {
                ACTOR: None,
                CRITIC: None,
            }
        batch[SampleBatch.SEQ_LENS] = None

        #batch['obs'] = self.blahencoder(batch['obs'].cuda())
        encoder_outs = self.encoder(batch)
        # TODO (Artur): Un-uncomment once Policy supports RNN
        # output[STATE_OUT] = encoder_outs[STATE_OUT]

        # Value head
        vf_out = self.vf(encoder_outs[ENCODER_OUT][CRITIC])
        output[SampleBatch.VF_PREDS] = vf_out.squeeze(-1)

        # Policy head
        action_logits = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits

        return output

spec = SingleAgentRLModuleSpec(
    module_class=PPOTorchRLModule,
    observation_space=gym.spaces.Box(-1, 1, (512,)),
    action_space=gym.spaces.Discrete(18),
    model_config_dict={"vf_share_layers": True,
        "fcnet_hiddens": [256, 256]}
)

config = (
    PPOConfig()
    .environment(CusEnv, env_config={
        "frameskip": 1,
        "full_action_space": False,
        "repeat_action_probability": 0.0
    })
    .training(_enable_learner_api=True)
    .rl_module(
        _enable_rl_module_api=True,
        rl_module_spec=spec,
    )
    .training(model={"vf_share_layers": True,
        "fcnet_hiddens": [256, 256],
        "conv_filters": [[16, 4, 2], [32, 4, 2], [64, 4, 2], [128, 4, 2]]},
        lambda_=0.95,
        lr=0.0001,
        kl_coeff=0.5,
        clip_param=0.1,
        vf_clip_param=10.0,
        grad_clip=100.0,
        entropy_coeff=0.01,
        train_batch_size=5000,
        sgd_minibatch_size=500,
        num_sgd_iter=10)
    #.exploration(exploration_config={})
    .reporting(min_time_s_per_iteration=30)
    .rollouts(num_rollout_workers=46,
        num_envs_per_worker=1,
        rollout_fragment_length='auto')
    .resources(
        num_gpus_per_learner_worker = .4,
        num_gpus=0,
        num_learner_workers=1,
        num_cpus_per_worker=0.2,
        num_gpus_per_worker=.14)
        #num_cpus_for_local_worker = 1,
)

algorithm = config.build()


# run for 2 training steps
for _ in range(2000000):
    result = algorithm.train()
    pprint(result)
