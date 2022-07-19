from sample_factory.runner.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333, 4444]),
        (
            "env",
            [
                "LunarLander-v2",
            ],
        ),
    ]
)

_experiments = [
    Experiment(
        "box2d_test_run",
        "python -m sf_examples.train_gym_env --train_for_env_steps=500000000 --algo=APPO --num_workers=12 --num_envs_per_worker=6 --seed 0 --gae_lambda 0.99 --experiment=lunar_lander_2 --env=gym_LunarLanderContinuous-v2 --exploration_loss_coeff=0.0 --max_grad_norm=0.0 --encoder_type=mlp --encoder_subtype=mlp_mujoco --encoder_extra_fc_layers=0 --hidden_size=128 --policy_initialization=xavier_uniform --actor_critic_share_weights=False --adaptive_stddev=False --recurrence=1 --use_rnn=False --batch_size=256 --num_epochs=4 --with_vtrace=False --reward_scale=0.05 --max_policy_lag=100000 --save_every_sec=15 --experiment_summaries_interval=10 --with_wandb=True --wandb_user=wmFrank --wandb_project=box2d-benchmark --wandb_group=lunarlandercont --wandb_tags run0",
        _params.generate_params(randomize=False),
    ),
]

RUN_DESCRIPTION = RunDescription("box2d_envs", experiments=_experiments)
# python -m sample_factory.runner.run --run=sf_examples.box2d_examples.box2d_evns --runner=processes --max_parallel=12  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1
