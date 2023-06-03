import argparse
import torch

#(args.backbone, args.setting, args.trainset, args.expname)
#args.expname only makes sense if args.setting is not allgame

#if args.backbone e2e.. then the gradients flow through the backbone during training
def get_args():
    parser = argparse.ArgumentParser(description='RL')
    #parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        choices=["e2e", "vae", "alloff", "eachmixedoff", "eachmediumoff", "eachexpertoff", "allmixedoff", "allmediumoff", "allexpertoff", "random", "imagenet", "voltron", "r3m", "value"],
        default="e2e",
    )
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test: --stop-reward must "
             "be achieved within --stop-timesteps AND --stop-iters.",
    )
    
    parser.add_argument(
        "--machine", type=str, default="", help="machine to be training"
    )
    parser.add_argument(
        "--log", type=str, default="/lab/kiran/logs/rllib/atari", help="config file for resources"
    )
    parser.add_argument(
        "--ckpt", type=str, default="/lab/kiran/ckpts/trained", help="directory for saving resources"
    ) 
    parser.add_argument(
        "--env_name", type=str, default="atari", help="Environment name"
    )
    parser.add_argument(
        "--set", type=str, choices=["all", "train", "test"], default="all", help="ALE/Pong-v5"
    )
    parser.add_argument(
        "--setting", type=str, choices=["eachgame", "seqgame", "allgame"], default="eachgame", help="Each game"
    )
    parser.add_argument(
        "--expname", type=str, choices=["backbone", "backbonepolicy", "full", "ours"], default="full", help="ALE/Pong-v5"
    )
    parser.add_argument(
        "--policy", type=str, default="PolicyNotLoaded", help="ALE/Pong-v5"
    )    
    parser.add_argument(
        "--temporal", type=str, choices=["attention", "lstm", "4stack"], default="4stack", help="temporal model"
    )
    parser.add_argument(
        "--prefix", type=str, default="1.a", help="which baseline is it"
    )
    parser.add_argument(
        "--train", action='store_true'
    )
    parser.add_argument(
        "--eval", action='store_true'
    )
    parser.add_argument(
        "--stop_timesteps", type=int, default=15000000, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--lambda_", type=float, default=.95, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--kl_coeff", type=float, default=.5, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--clip_param", type=float, default=.1, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--entropy_coeff", type=float, default=.01, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--gamma", type=float, default=.95, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--vf_clip", type=float, default=10, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--buffer_size", type=int, default=5000, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--batch_size", type=int, default=500, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--num_epoch", type=int, default=10, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of GPUs each worker has"
    )
    
    parser.add_argument(
        "--num_envs", type=int, default=5, help="Number of envs each worker evaluates"
    )
    
    parser.add_argument(
        "--num_gpus", type=float, default=.2, help="Number of GPUs each worker has"
    )

    parser.add_argument(
        "--gpus_worker", type=float, default=.1, help="Number of GPUs each worker has"
    ) 

    parser.add_argument(
        "--cpus_worker", type=float, default=.5, help="Number of CPUs each worker has"
    )

    #use_lstm or framestacking
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run with/without Tune using a manual train loop instead. If ran without tune, use PPO without grid search and no TensorBoard.",
    )

    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )

    args = parser.parse_args()

    return args