from arguments import get_args
from ray.tune.logger import pretty_print, UnifiedLogger, Logger, LegacyLoggerCallback

import os

args = get_args()

resource_file = '/lab/kiran/hostfile.yaml'

#pathnames for all the saved .pth backbonemodels
mapfile =  {"vae": "/lab/kiran/ckpt/vae", "e2e": "/lab/kiran/ckpt/e2e"}

#add the model to a mapfile dictionary

all_atari_envs = ["AirRaidNoFrameskip-v4","AssaultNoFrameskip-v4","BeamRiderNoFrameskip-v4", "CarnivalNoFrameskip-v4","DemonAttackNoFrameskip-v4","NameThisGameNoFrameskip-v4","PhoenixNoFrameskip-v4","RiverraidNoFrameskip-v4","SpaceInvadersNoFrameskip-v4"]
train_atari_envs = ["AirRaidNoFrameskip-v4","AssaultNoFrameskip-v4","BeamRiderNoFrameskip-v4", "DemonAttackNoFrameskip-v4","RiverraidNoFrameskip-v4","SpaceInvadersNoFrameskip-v4"]
test_atari_envs = ["CarnivalNoFrameskip-v4", "NameThisGameNoFrameskip-v4", "PhoenixNoFrameskip-v4"]


#all_beogym_envs = []
#train_beogym_envs = []
#test_beogym_envs = []


#atari env -> multiple games
#beogym env -> multiple cities
#carla env -> multiple towns


atari_config = {
    "env" : args.env_name,
    "clip_rewards" : True,
    "framework" : "torch",
    "logger_config": {
        "type": UnifiedLogger,
        "logdir": os.path.expanduser(args.log)
        },
    "observation_filter":"NoFilter",
    "num_workers": args.num_workers,
    "rollout_fragment_length" : 100,
    "num_envs_per_worker" : args.num_envs,
    "model":{
            "vf_share_layers" : True,

    },
    #"lambda_" : args.lambda_,
    "kl_coeff" : args.kl_coeff,
    "clip_param" : args.clip_param,
    "entropy_coeff" : args.entropy_coeff,
    "gamma" : args.gamma,
    "vf_clip_param" : args.vf_clip,
    "train_batch_size":args.buffer_size,
    "sgd_minibatch_size":args.batch_size,
    "num_sgd_iter":args.num_epoch,
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus":args.num_gpus,
    "num_gpus_per_worker" : args.gpus_worker,
    "num_cpus_per_worker":args.cpus_worker
    }

"""
create beogym_config from atari_config
beogym_config = {
    "env" : args.env_name,
    "clip_rewards" : True,
    "framework" : "torch",
    "logger_config": {
        "type": UnifiedLogger,
        "logdir": os.path.expanduser(args.log)
        },
    "observation_filter":"NoFilter",
    "num_workers":args.num_workers,
    "rollout_fragment_length" : args.roll_frags,
    "num_envs_per_worker" : args.num_envs,
    "model":{
            "vf_share_layers" : True,

    },
    #"lambda_" : args.lambda_,
    "kl_coeff" : args.kl_coeff,
    "clip_param" : args.clip_param,
    "entropy_coeff" : args.entropy_coeff,
    "gamma" : args.gamma,
    "vf_clip_param" : args.vf_clip,
    "train_batch_size":args.buffer_size,
    "sgd_minibatch_size":args.batch_size,
    "num_sgd_iter":args.num_epoch,
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus":args.num_gpus,
    "num_gpus_per_worker" : args.gpus_worker,
    "num_cpus_per_worker":args.cpus_worker
    }




hyperparam_mutations = {
    "lambda": lambda: random.uniform(0.9, 1.0),
    "clip_param": lambda: random.uniform(0.01, 0.5),
    "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    "num_sgd_iter": lambda: random.randint(1, 30),
    "sgd_minibatch_size": lambda: random.randint(128, 16384),
    "train_batch_size": lambda: random.randint(2000, 160000),
}

"""