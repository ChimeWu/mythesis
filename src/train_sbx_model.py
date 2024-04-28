from sbx import DDPG, PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import pandas as pd
from os.path import abspath, dirname, join

from env import RyeFlexEnvFixed


def train_sbx_ddpg():
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnvFixed(data)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=36000)
    model.save("sbx_ddpg_rye_flex_env")

def train_sbx_ppo():
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnvFixed(data)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=36000)
    model.save("sbx_ppo_rye_flex_env")

def train_sbx_sac():
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnvFixed(data)

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=36000)
    model.save("sbx_sac_rye_flex_env")

if __name__ == "__main__":
    train_sbx_ddpg()
    train_sbx_ppo()
    train_sbx_sac()