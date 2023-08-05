def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

pbt_hyperparam_mutations = {
    "lambda": lambda: random.uniform(0.9, 1.0),
    "clip_param": lambda: random.uniform(0.01, 0.5),
    "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    "num_sgd_iter": lambda: random.randint(1, 30),
    "sgd_minibatch_size": lambda: random.randint(128, 16384),
    "train_batch_size": lambda: random.randint(2000, 160000),
    }

pb2_hyperparam_mutations = {
    "lambda": [0.9, 1.0],
    "clip_param": [0.01, 0.5],
    "lr": [1e-3, 1e-5],
    "num_sgd_iter": [1, 30],
    "sgd_minibatch_size": [128, 16384],
    "train_batch_size": [2000, 160000],
    }

pbt = PopulationBasedTraining(
    time_attr="time_total_s",
    perturbation_interval=120,
    resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations=pbt_hyperparam_mutations,
    custom_explore_fn=explore,
    )

pb2 = pb2.PB2(
    time_attr="time_total_s",
    perturbation_interval=50000,
    quantile_fraction=0.25,  # copy bottom % with top % (weights)
    # Specifies the hyperparam search space
    hyperparam_bounds=pb2_hyperparam_mutations
    )



    """
    else:

        tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=pbt,
            num_samples=4,
        ),
        param_space=config,
        run_config=air.RunConfig(stop={
            "timesteps_total": args.stop_timesteps,
            }),
        )
        results = tuner.fit()

"""