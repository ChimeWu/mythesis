import pandas as pd
from os.path import abspath, dirname, join
from datetime import datetime
from sbx import DDPG, PPO, SAC

from env import *

def test_sbx_ddpg():
    plotter = RyeFlexEnvEpisodePlotter()

    # 加载并运行模型
    model = DDPG.load("sbx_ddpg_rye_flex_env")
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/test.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)
    state = env.reset(start_time=datetime(2021, 2, 1, 0, 0))
    info = {}
    done = False

    while not done:
        action = model.predict(state)[0]

        state, reward, done, info = env.step(action)

        plotter.update(info)

    print(f"Your score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode()

def test_sbx_ppo():
    plotter = RyeFlexEnvEpisodePlotter()

    # 加载并运行模型
    model = PPO.load("sbx_ppo_rye_flex_env")
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/test.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)
    state = env.reset(start_time=datetime(2021, 2, 1, 0, 0))
    info = {}
    done = False

    while not done:
        action = model.predict(state)[0]

        state, reward, done, info = env.step(action)

        plotter.update(info)

    print(f"Your score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode()

def test_sbx_sac():
    plotter = RyeFlexEnvEpisodePlotter()

    # 加载并运行模型
    model = SAC.load("sbx_sac_rye_flex_env")
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/test.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)
    state = env.reset(start_time=datetime(2021, 2, 1, 0, 0))
    info = {}
    done = False

    while not done:
        action = model.predict(state)[0]

        state, reward, done, info = env.step(action)

        plotter.update(info)

    print(f"Your score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode()

if __name__ == "__main__":
    test_sbx_ddpg()
    test_sbx_ppo()
    test_sbx_sac()