def atari_override_defaults(env, parser):
    """RL params specific to Atari envs."""
    parser.set_defaults(
        encoder_type="mlp",
        encoder_subtype="mlp_cc",
        hidden_size=64,
        obs_subtract_mean=0.0,
        obs_scale=1.0,
        gamma=0.99,
        # reward_clip=1.0,  # same as APE-X paper
        # env_frameskip=4,
        # env_framestack=4,
        exploration_loss_coeff=0.01,
    )


def atari_benchmark_defaults(env, parser):
    parser.set_defaults(
        # multiply_frameskip=False,
        encoder_type="mlp",
        encoder_subtype="mlp_cc",
        # actor_critic_share_weights=False,
        hidden_size=64,
        encoder_extra_fc_layers=0,
        obs_subtract_mean=0.0,
        obs_scale=1.0,
        gamma=0.99,
        # reward_clip=1.0,  # same as APE-X paper
        # env_frameskip=4,
        # env_framestack=4,
        exploration_loss_coeff=0.01,
        num_workers=4,
        num_envs_per_worker=1,
        worker_num_splits=1,
        train_for_env_steps=500000,
        nonlinearity="tanh",
        kl_loss_coeff=0.0,
        use_rnn=False,
        adaptive_stddev=False,
        # policy_initialization="torch_default",
        reward_scale=1.0,
        with_vtrace=False,
        recurrence=1,
        batch_size=128,
        rollout=128,
        max_grad_norm=0.5,
        num_epochs=4,
        num_batches_per_epoch=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=0.5,
        exploration_loss="entropy",
        learning_rate=0.00025,
        lr_schedule="linear_decay",
        shuffle_minibatches=True,
        gae_lambda=0.95,
        batched_sampling=False,
        normalize_input=False,
        normalize_returns=False,
        serial_mode=False,
        async_rl=False,
        experiment_summaries_interval=3,
        adam_eps=1e-5,
    )


# noinspection PyUnusedLocal
def add_atari_env_args(env, parser):
    # in case we more args in the future
    pass
