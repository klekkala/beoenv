from arguments import get_args
from ray.tune.logger import pretty_print, UnifiedLogger, Logger, LegacyLoggerCallback
import os

args = get_args()




resource_file = '/lab/kiran/hostconf/'

#pathnames for all the saved .pth backbonemodels
#IMPLEMENT VAE FOR BEOGYM
#map_models =  {"1chanvae": "/lab/kiran/ckpts/pretrained/atari/GREY_ATARI_ALL_0.0_128.pt", "4stackvae": "/lab/kiran/ckpts/pretrained/atari/4STACK_ATARI_BEAMRIDER_0.0_128.pt", "random": None, "e2e": None}

#add the model to a mapfile dictionary

if args.env_name == "atari":
    if args.set == "all":
        all_envs = ["AirRaidNoFrameskip-v4","AssaultNoFrameskip-v4","BeamRiderNoFrameskip-v4", "CarnivalNoFrameskip-v4","DemonAttackNoFrameskip-v4","NameThisGameNoFrameskip-v4","PhoenixNoFrameskip-v4","RiverraidNoFrameskip-v4","SpaceInvadersNoFrameskip-v4"]
        #all_envs = ["BeamRiderNoFrameskip-v4", "AssaultNoFrameskip-v4"]
    elif args.set == "train": 
        all_envs = ["AirRaidNoFrameskip-v4", "CarnivalNoFrameskip-v4", "DemonAttackNoFrameskip-v4", "NameThisGameNoFrameskip-v4" ,"SpaceInvadersNoFrameskip-v4"]
    elif args.set == "test":
        all_envs = ["AssaultNoFrameskip-v4", "BeamRiderNoFrameskip-v4", "RiverraidNoFrameskip-v4", "PhoenixNoFrameskip-v4"]
    elif args.set == "one":
        all_envs = ["AirRaidNoFrameskip-v4"]
    elif args.set == "two":
        all_envs = ["AirRaidNoFrameskip-v4", "CarnivalNoFrameskip-v4"]
    elif args.set == "three":
        all_envs = ["AirRaidNoFrameskip-v4", "CarnivalNoFrameskip-v4", "DemonAttackNoFrameskip-v4"]
    elif args.set == "four":
        all_envs = ["AirRaidNoFrameskip-v4", "CarnivalNoFrameskip-v4", "DemonAttackNoFrameskip-v4", "NameThisGameNoFrameskip-v4"]


elif args.env_name == "beogym":
    print("lksjfdlkskjfkalsj;fdjkfalsjfdkljfl")
    if args.set == "all":
        all_envs = ['Wall_Street', 'Union_Square', 'Hudson_River', 'CMU', 'Allegheny', 'South_Shore']
    elif args.set == "train": 
        all_envs = ['Wall_Street', 'Union_Square', 'Hudson_River', 'CMU', 'Allegheny']
    elif args.set == "test":
        all_envs = ['South_Shore']
    elif args.set == "one":
        all_envs = ['Wall_Street']
    elif args.set == "two":
        all_envs = ['Wall_Street', 'Union_Square']
    elif args.set == "three":
        all_envs = ['Wall_Street', 'Union_Square', 'Hudson_River']
    elif args.set == "four":
        all_envs = ['Wall_Street', 'Union_Square', 'Hudson_River', 'CMU']



#atari env -> multiple games
#beogym env -> multiple cities
#carla env -> multiple towns

if args.backbone == "e2e":
    args.train_backbone = True

#if args.set == "CarnivalNoFrameskip-v4":
#    args.horizon = 475

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
        "custom_model_config" : {"backbone": args.backbone, "backbone_path": args.ckpt + args.env_name + "/" + args.backbone, "train_backbone": args.train_backbone, 'temporal': args.temporal, "div": args.div},
        "framestack": args.temporal == '4stack',
        "use_lstm": args.temporal == 'lstm',
        "use_attention": args.temporal == 'attention',
    },
    "horizon": args.horizon,
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
    "num_workers":args.num_workers-8,
    "rollout_fragment_length" : 1000,
    "num_envs_per_worker" : args.num_envs-8,
    'model':{
                "custom_model": "model",
                "use_lstm": True,
                "lstm_cell_size": 256,
                "lstm_use_prev_action" : True,
                "lstm_use_prev_reward" : True,
                "vf_share_layers": True,
                #"conv_filters": [[16, 3, 2], [32, 3, 2], [64, 3, 2], [128, 3, 2], [512, 3, 2]],
                "conv_filters": [[16, [8, 8], 4], [32, [4, 4], 2], [512, [11, 11], 1]],
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

colo_config = {
    "env" : args.env_name,
    "framework" : "torch",
    "logger_config": {
        "type": UnifiedLogger,
        "logdir": os.path.expanduser(args.log)
        },
    "observation_filter":"NoFilter",
    "num_workers": args.num_workers,
    "rollout_fragment_length" : 1000,
    "num_envs_per_worker" : 1,
    'model':{
                "custom_model": "model",
                "vf_share_layers": True,
                "conv_filters": [[16, [8, 8], 4], [32, [4, 4], 2], [512, [11, 11], 1]],
                "conv_activation":'relu',
                "post_fcnet_hiddens":[],
            },
    "kl_coeff" : args.kl_coeff,
    "clip_param" : 0.1,
    "entropy_coeff" : 0.01,
    "vf_clip_param" : 10.0,
    "train_batch_size":20000,
    "sgd_minibatch_size":2000,
    "num_sgd_iter": 10,
    "lr" : args.lr,
    "num_gpus":args.num_gpus,
    "num_gpus_per_worker" : args.gpus_worker,
    "num_cpus_per_worker":args.cpus_worker
    }
