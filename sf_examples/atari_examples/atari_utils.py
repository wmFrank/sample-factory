import gym

from sample_factory.envs.env_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    FrameStack,
    MaxAndSkipEnv,
    NoopResetEnv,
    PixelFormatChwWrapper,
    ResizeWrapper,
    SkipAndStackFramesWrapper,
    SkipFramesWrapper,
    StackFramesWrapper,
)

ATARI_W = ATARI_H = 84


class AtariSpec:
    def __init__(self, name, env_id, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout
        self.has_timer = False


ATARI_ENVS = [
    AtariSpec("atari_montezuma", "MontezumaRevengeNoFrameskip-v4", default_timeout=18000),
    AtariSpec("atari_pong", "PongNoFrameskip-v4"),
    AtariSpec("atari_qbert", "QbertNoFrameskip-v4"),
    AtariSpec("atari_breakout", "BreakoutNoFrameskip-v4"),
    AtariSpec("atari_spaceinvaders", "SpaceInvadersNoFrameskip-v4"),
    AtariSpec("atari_asteroids", "AsteroidsNoFrameskip-v4"),
    AtariSpec("atari_gravitar", "GravitarNoFrameskip-v4"),
    AtariSpec("atari_mspacman", "MsPacmanNoFrameskip-v4"),
    AtariSpec("atari_seaquest", "SeaquestNoFrameskip-v4"),
    AtariSpec("atari_beamrider", "BeamRiderNoFrameskip-v4"),
    AtariSpec("atari_cartpole", "CartPole-v1"),
    AtariSpec("atari_acrobot", "Acrobot-v1"),
    AtariSpec("atari_mountaincar", "MountainCar-v0"),
]


def atari_env_by_name(name):
    for cfg in ATARI_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Atari env")


def make_atari_env(env_name, cfg, env_config, **kwargs):
    atari_spec = atari_env_by_name(env_name)
    env = gym.make(atari_spec.env_id)
    return env
