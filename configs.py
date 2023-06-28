from arguments import get_args
from ray.tune.logger import pretty_print, UnifiedLogger, Logger, LegacyLoggerCallback
import os

args = get_args()




resource_file = '/lab/kiran/hostconf/'

#pathnames for all the saved .pth backbonemodels
#IMPLEMENT VAE FOR BEOGYM
map_models =  {"1chanvae": "/lab/kiran/ckpts/pretrained/atari/GREY_ATARI_BEAMRIDER_0.0_64.pt", "4stackvae": "/lab/kiran/ckpts/pretrained/atari/4STACK_ATARI_BEAMRIDER_0.0_64.pt", "random": None, "e2e": None}

#add the model to a mapfile dictionary

if args.env_name == "atari":
    if args.set == "all":
        all_envs = ["AirRaidNoFrameskip-v4","AssaultNoFrameskip-v4","BeamRiderNoFrameskip-v4", "CarnivalNoFrameskip-v4","DemonAttackNoFrameskip-v4","NameThisGameNoFrameskip-v4","PhoenixNoFrameskip-v4","RiverraidNoFrameskip-v4","SpaceInvadersNoFrameskip-v4"]
        #all_envs = ["BeamRiderNoFrameskip-v4", "AssaultNoFrameskip-v4"]
    elif args.set == "train": 
        all_envs = ["AirRaidNoFrameskip-v4", "CarnivalNoFrameskip-v4", "DemonAttackNoFrameskip-v4", "NameThisGameNoFrameskip-v4" ,"SpaceInvadersNoFrameskip-v4"]
    else:
        all_envs = ["AssaultNoFrameskip-v4", "BeamRiderNoFrameskip-v4", "RiverraidNoFrameskip-v4", "PhoenixNoFrameskip-v4"]

elif args.env_name == "beogym":
    all_envs = ['Wall_Street', 'Union_Square', 'Hudson_River']
    #train_beogym_envs = []
    #test_beogym_envs = []


#atari env -> multiple games
#beogym env -> multiple cities
#carla env -> multiple towns

if args.backbone == "e2e":
    args.train_backbone = True

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
    "model": {
        "custom_model": "model",
        "vf_share_layers": True,
        "conv_filters": [[16, [8, 8], 4], [32, [4, 4], 2], [512, [11, 11], 1],],
        "conv_activation" : "relu" if (args.temporal == '4stack' or args.temporal == 'notemp') else "elu",
        "custom_model_config" : {"backbone": args.backbone, "backbone_path": map_models[args.backbone], "train_backbone": args.train_backbone, 'temporal': args.temporal},
        "framestack": args.temporal == '4stack',
        "use_lstm": args.temporal == 'lstm',
        "use_attention": args.temporal == 'attention',
    },
    "kl_coeff" : args.kl_coeff,
    "clip_param" : args.clip_param,
    "entropy_coeff" : args.entropy_coeff,
    "gamma" : args.gamma,
    "lr" : args.lr,
    "vf_clip_param" : args.vf_clip,
    "train_batch_size":args.buffer_size,
    "sgd_minibatch_size":args.batch_size,
    "num_sgd_iter":args.num_epoch,
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus":args.num_gpus,
    "num_gpus_per_worker" : args.gpus_worker,
    "num_cpus_per_worker":args.cpus_worker
    }


beogym_config = {
    "env" : args.env_name,
    "framework" : "torch",
    "logger_config": {
        "type": UnifiedLogger,
        "logdir": os.path.expanduser(args.log)
        },
    "observation_filter":"NoFilter",
    "num_workers":args.num_workers,
    "rollout_fragment_length" : 1000,
    "num_envs_per_worker" : args.num_envs,
    'model':{
                "use_lstm": True,
                "lstm_cell_size": 256,
                "lstm_use_prev_action" : True,
                "lstm_use_prev_reward" : True,
                "vf_share_layers": True,
                "conv_filters": [[16, 3, 2], [32, 3, 2], [64, 3, 2], [128, 3, 2], [256, 3, 2]],
                "conv_activation":'relu',
                "post_fcnet_hiddens":[],
            },
    "kl_coeff" : 0.5,
    "clip_param" : 0.1,
    "entropy_coeff" : 0.01,
    "vf_clip_param" : 10.0,
    "train_batch_size":20000,
    "sgd_minibatch_size":2000,
    "num_sgd_iter": 10,
    "num_gpus":args.num_gpus,
    "num_gpus_per_worker" : args.gpus_worker,
    "num_cpus_per_worker":args.cpus_worker
    }


