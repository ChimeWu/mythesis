from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import pandas as pd
from os.path import abspath, dirname, join
from datetime import datetime

from env import *

        
def testmodel():
    plotter = RyeFlexEnvEpisodePlotter()

    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/test.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)
    state = env.reset(start_time=datetime(2021, 2, 1, 0, 0))
    model = SimpleStateBasedAgent(env.action_space)
    info = {}
    done = False

    while not done:
        action = model.get_action(state)

        state, reward, done, info = env.step(action)

        plotter.update(info)

    print(f"Your score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode()


def testmodel0():
    plotter = RyeFlexEnvEpisodePlotter()

    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/test.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)
    state = env.reset(start_time=datetime(2021, 2, 1, 0, 0))

    model = RandomActionAgent(env.action_space)
    info = {}
    done = False


    while not done:
        action = model.get_action()
        state, reward, done, info = env.step(action)
        plotter.update(info)

    print(f"Your score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode()

def testmodel1():
    plotter = RyeFlexEnvEpisodePlotter()

    # 加载并运行模型
    model = PPO.load("ppo_rye_flex_env")
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

def testmodel2():
    plotter = RyeFlexEnvEpisodePlotter()

    # 加载并运行模型
    model = A2C.load("a2c_rye_flex_env")
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

def testmodel3():
    plotter = RyeFlexEnvEpisodePlotter()

    # 加载并运行模型
    model = DDPG.load("ddpg_rye_flex_env")
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

def testmodelusetrain1():
    plotter = RyeFlexEnvEpisodePlotter()

    # 加载并运行模型
    model = PPO.load("ppo_rye_flex_env")
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)
    state = env.reset(start_time=datetime(2020, 2, 1, 0, 0))
    info = {}
    done = False

    while not done:
        action = model.predict(state,deterministic=True)[0]

        state, reward, done, info = env.step(action)

        plotter.update(info)

    print(f"Your score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode()

def testmodelusetrain2():
    plotter = RyeFlexEnvEpisodePlotter()

    # 加载并运行模型
    model = A2C.load("a2c_rye_flex_env")
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)
    state = env.reset(start_time=datetime(2020, 2, 1, 0, 0))
    info = {}
    done = False

    while not done:
        action = model.predict(state,deterministic=True)[0]

        state, reward, done, info = env.step(action)

        plotter.update(info)

    print(f"Your score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode()

def testmodelusetrain3():
    plotter = RyeFlexEnvEpisodePlotter()

    # 加载并运行模型
    model = DDPG.load("ddpg_rye_flex_env")
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)
    state = env.reset(start_time=datetime(2020, 2, 1, 0, 0))
    info = {}
    done = False

    while not done:
        action = model.predict(state,deterministic=True)[0]

        state, reward, done, info = env.step(action)

        plotter.update(info)

    print(f"Your score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode()


if __name__ == "__main__":
    testmodel()
    testmodel0()
    testmodel1()
    testmodel2()
    testmodel3()
    testmodelusetrain1()
    testmodelusetrain2()
    testmodelusetrain3()