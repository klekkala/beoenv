import graph_tool.all as gt
import yaml
from ray.tune.schedulers import PopulationBasedTraining
import sys
import time
import argparse
from pathlib import Path
import ray
import os
from ray.tune.logger import pretty_print, UnifiedLogger, Logger, LegacyLoggerCallback
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.annotations import override
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from Zero import ZeroNetwork as TorchZero
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
from beogym import BeoGym




if __name__ == "__main__":

    r = 128
    steps = 100


    torch, nn = try_import_torch()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
    )
    parser.add_argument(
        "--framework",
        choices=["torch"],
        default="torch",
    )
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test: --stop-reward must "
             "be achieved within --stop-timesteps AND --stop-iters.",
    )
    parser.add_argument(
        "--stop-iters", type=int, default=100, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps", type=int, default=2000000, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--machine", type=str, default="None", help="machine to be training"
    )
    parser.add_argument(
        "--config_file", type=str, default="/lab/kiran/beoenv/hostfile.yaml", help="config file for resources"
    )

    parser.add_argument(
        "--config", type=str, default="/lab/kiran/beoenv/hostfile.yaml", help="config file for resources"
    )
    
    parser.add_argument(
        "--log", type=str, default="/lab/kiran/logs/rllib/beogym", help="config file for resources"
    )

    parser.add_argument(
        "--num_workers", type=int, default=20, help="Number of GPUs each worker has"
    )

    parser.add_argument(
        "--num_envs", type=int, default=2, help="Number of envs each worker evaluates"
    )

    parser.add_argument(
        "--roll_frags", type=int, default=100, help="Rollout fragments"
    )

    parser.add_argument(
        "--num_gpus", type=float, default=1, help="Number of GPUs each worker has"
    )

    parser.add_argument(
        "--gpus_worker", type=float, default=.3, help="Number of GPUs each worker has"
    )

    parser.add_argument(
        "--cpus_worker", type=float, default=.5, help="Number of CPUs each worker has"
    )

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
    parser.add_argument(
        "--no_image",
        choices=[False,True],
        default=False,
    )


    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    class TorchCustomModel(TorchModelV2, nn.Module):
        """Example of a PyTorch custom model that just delegates to a fc-net."""

        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
            nn.Module.__init__(self)

            self.torch_sub_model = TorchZero(
                obs_space, action_space, num_outputs, model_config, name
            )

        # @profile(precision=5)
        def forward(self, input_dict, state, seq_lens):
            # input_dict["obs"]["obs"] = input_dict["obs"]["obs"].float()
            fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
            return fc_out, []

        def value_function(self):
            return torch.reshape(self.torch_sub_model.value_function(), [-1])

    class TorchNoImageModel(TorchModelV2, nn.Module):
        """Example of a PyTorch custom model that just delegates to a fc-net."""

        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
            nn.Module.__init__(self)

            self.torch_sub_model = TorchFC(
                obs_space, action_space, num_outputs, model_config, name
            )

        # @profile(precision=5)
        def forward(self, input_dict, state, seq_lens):
            input_dict["obs"] = input_dict["obs"].float()
            fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
            return fc_out, []

        def value_function(self):
            return torch.reshape(self.torch_sub_model.value_function(), [-1])

    class MyPrintLogger(Logger):
        """Logs results by simply printing out everything."""
        def _init(self):
            # Custom init function.
            print("Initializing ...")
            # Setting up our log-line prefix.
            self.prefix = self.config.get("logger_config").get("prefix")
        def on_result(self, result: dict):
            # Define, what should happen on receiving a `result` (dict).
            print(f"{self.prefix}: {result}")
        def close(self):
            # Releases all resources used by this logger.
            print("Closing")
        def flush(self):
            # Flushing all possible disk writes to permanent storage.
            print("Flushing ;)", flush=True)

    args = parser.parse_args()
    if args.tune:
        args.config_file = '/lab/kiran/BeoEnv/tune.yaml'

        # extract data from the config file
    if args.machine is not None:
        with open(args.config_file, 'r') as cfile:
            config_data = yaml.safe_load(cfile)

    #args.num_workers, args.num_envs, args.num_gpus, args.gpus_worker, args.cpus_worker, args.roll_frags = config_data[args.machine]
    ray.init(local_mode=args.local_mode)
    if args.no_image:
        ModelCatalog.register_custom_model(
            "my_model", TorchNoImageModel
        )
    else:
        ModelCatalog.register_custom_model(
            "my_model", TorchCustomModel
        )

    str_logger = str(args.stop_timesteps)+str(time.time())
    os.mkdir(os.path.expanduser(args.log) + '/' + str_logger)
    config = (
        PPOConfig()
            .environment(BeoGym, env_config = {"no_image":args.no_image})
            .framework("torch")
            .rollouts(num_rollout_workers=args.num_workers,
                      rollout_fragment_length='auto',
                      num_envs_per_worker=args.num_envs,
                      ignore_worker_failures=True)
            .training(
            model={
                "custom_model": "my_model",
                "vf_share_layers": True,
                "conv_filters": [[16, [3, 5], [1,2]], [32, [5, 5], 2], [64, [5, 5], 3], [128, [5, 5], 4], [256, [9, 9], 1]],
                "post_fcnet_hiddens": [64],
            },
            lambda_=0.95,
            kl_coeff=0.5,
            clip_param=0.1,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            train_batch_size=2000,
            sgd_minibatch_size=200,
            num_sgd_iter=10,

        )
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            .resources(num_gpus=7
                , num_gpus_per_worker=.33,
                       num_cpus_per_worker=.5
                       )
            .debugging(logger_config={
            "type": UnifiedLogger,
            "logdir": os.path.expanduser(args.log) + '/' + str_logger
            }
                )
    )




    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
    }

    if args.tune == False:
        start=time.time()
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        # config.lr = 5e-4
        algo = config.build()
        # run manual training loop and print results after each iteration
        for _ in range(100000000):
            result = algo.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if result["timesteps_total"] >= args.stop_timesteps:
                path_to_checkpoint = algo.save()
                print(path_to_checkpoint)
                print(time.time()-start)
                break
        algo.stop()
    else:
        hyperparam_mutations = {
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(2000, 160000),
        }

        pbt = PopulationBasedTraining(
            time_attr="time_total_s",
            perturbation_interval=120,
            resample_probability=0.25,
            # Specifies the mutations of these hyperparams
            hyperparam_mutations=hyperparam_mutations,
            custom_explore_fn=explore,
        )

        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            args.run,
            tune_config=tune.TuneConfig(
                metric="episode_reward_mean",
                mode="max",
                scheduler=pbt,
                num_samples=2,
            ),
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop),
        )
        results = tuner.fit()

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)


 
