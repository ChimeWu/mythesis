from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import pandas as pd
from os.path import abspath, dirname, join

from env import RyeFlexEnvFixed

def train_ddpg():
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnvFixed(data)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=3600)
    model.save("ddpg_rye_flex_env")

def train_ppo():
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnvFixed(data)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=3600)
    model.save("ppo_rye_flex_env")

def train_a2c():
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnvFixed(data)

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=3600)
    model.save("a2c_rye_flex_env")


def main():
    train_ddpg()
    train_ppo()
    train_a2c()

if __name__ == "__main__":
    main()