import logging
from datetime import datetime, timedelta
from random import randrange, seed
from typing import Any, Dict, List, Optional, Tuple, cast, Union
from dataclasses import dataclass
from gym import spaces

import matplotlib.pyplot as plt
import gym
import numpy as np
import pandas as pd

@dataclass
class Action:
    """The action vector.

    Args:
        charge_battery [kWh/h]: If value is:
            - Positive -> store energy in battery.
            - Negative -> discharged energy from battery and use in microgrid.
        charge_hydrogen [kWh/h]: If value is:
            - Positive -> store power in hydrogen.
            - Negative -> discharged from battery and use in microgrid.
    """

    charge_battery: float
    charge_hydrogen: float

    @property
    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.charge_battery,
                self.charge_hydrogen,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, action: np.ndarray) -> "Action":
        return cls(
            charge_battery=action.item(0),
            charge_hydrogen=action.item(1),
        )


@dataclass
class State:
    """The state vector.

    Args:
        consumption [kWh/h]: Consumption in the microgrid.
        wind_production [kWh/h]: Power production from solar used in the microgrid.
        pv_production [kWh/h]: Power production from solar wind used in the microgrid.
        battery_storage [kWh]: Potential energy stored in the battery.
        hydrogen_storage [kWh]: Potential energy stored in the hydrogen.
        grid_import [kWh/h]: Power imported from the grid to the microgrid.
        grid_import_peak [kWh/h]: Peak power imported from the grid to the microgrid.
        spot_market_price [NOK/kWh]: The spot market price for Trondheim.
    """



    """
    consumption [kWh/h]：微电网的消耗量。
    wind_production [kWh/h]：微电网使用的风力发电量。
    pv_production [kWh/h]：微电网使用的太阳能发电量。
    battery_storage [kWh]：储存在电池中的潜在能量。
    hydrogen_storage [kWh]：储存在氢气中的潜在能量。
    grid_import [kWh/h]：从电网向微电网输入的电力。
    grid_import_peak [kWh/h]：从电网向微电网输入的峰值电力。
    spot_market_price [NOK/kWh]：特隆赫姆的现货市场价格。
    
    """

    consumption: float
    pv_production: float
    wind_production: float
    battery_storage: float
    hydrogen_storage: float
    grid_import: float
    grid_import_peak: float
    spot_market_price: float

    @property
    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.consumption,
                self.pv_production,
                self.wind_production,
                self.battery_storage,
                self.hydrogen_storage,
                self.grid_import,
                self.grid_import_peak,
                self.spot_market_price,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, state: np.ndarray) -> "State":
        return cls(
            consumption=state.item(0),
            pv_production=state.item(1),
            wind_production=state.item(2),
            battery_storage=state.item(3),
            hydrogen_storage=state.item(4),
            grid_import=state.item(5),
            grid_import_peak=state.item(6),
            spot_market_price=state.item(7),
        )


logger = logging.getLogger("rye-flex-env")


def _get_hour_resolution(date: datetime) -> datetime:
    return datetime(date.year, date.month, date.day, date.hour)


class RyeFlexEnv(gym.Env):
    """RyeFlexGym is a simulator of the microgrid dynamics at Rye.

    Look at description-text for further explanation of this.

    Explanation of the state and action vector is found in states.py

    Attributes:
        _state: The current state of the system.
        _cumulative_reward: The cumulative reward from one episode.
        _time: The current time.
        _episode_length: The length of an episode.
        _time_resolution: The time between each step in an episode.
        _charge_loss_battery_storage: The charging loss in the battery storage.
        _charge_loss_hydrogen_storage: The charging loss in the hydrogen storage.
        _grid_tariff: The cost of energy imported from the grid.
        _peak_grid_tariff: The cost of the peak power imported from the grid.
        _action_space_min: Lower bound of action space.
        _action_space_max: Upper bound of action space.
        _state_space_min: Lower bound of state-space.
        _state_space_max: Upper bound of state-space.
        action_space: The action space. A requirement by OpenAiGym.
        observation_space: The observation space (which equal state space in this
            environment). A requirement by OpenAiGym.
        _measured_consumption_data: Historical data with measured consumption
            in the microgrid.
        _measured_pv_production_data: Historical data with measured solar production
            in the microgrid.
        _measured_wind_production_data: Historical data with measured wind production
            in the microgrid.
        _spot_market_price_data: Historical data of spot_market_price.
        _start_date_data: Start date of the historical data.
        _end_date_data: End date of the historical data.
        _episode_end_time: End date of episode.
        metadata: Data about which render modes exist for this env.
    """


    """
        _state：系统的当前状态。
        _cumulative_reward：一个情节的累积奖励。
        _time：当前时间。
        _episode_length：一个情节的长度。
        _time_resolution：情节中每一步的时间间隔。
        _charge_loss_battery_storage：电池储能的充电损失。
        _charge_loss_hydrogen_storage：氢储能的充电损失。
        _grid_tariff：从电网导入能源的成本。
        _peak_grid_tariff：从电网导入峰值电力的成本。
        _action_space_min：动作空间的下界。
        _action_space_max：动作空间的上界。
        _state_space_min：状态空间的下界。
        _state_space_max：状态空间的上界。
        action_space：动作空间。这是OpenAiGym的要求。
        observation_space：观察空间（在这个环境中等于状态空间）。这是OpenAiGym的要求。
        _measured_consumption_data：微电网中测量到的消耗的历史数据。
        _measured_pv_production_data：微电网中测量到的太阳能生产的历史数据。
        _measured_wind_production_data：微电网中测量到的风能生产的历史数据。
        _spot_market_price_data：现货市场价格的历史数据。
        _start_date_data：历史数据的开始日期。
        _end_date_data：历史数据的结束日期。
        _episode_end_time：情节的结束日期。
        metadata：关于这个环境存在哪些渲染模式的数据。
    """

    _state: State
    _cumulative_reward: float
    _time: datetime

    _episode_length: timedelta
    _time_resolution: timedelta
    _charge_loss_battery_storage: float
    _charge_loss_hydrogen_storage: float
    _grid_tariff: float
    _peak_grid_tariff: float

    _action_space_min: Action
    _action_space_max: Action
    _state_space_min: State
    _state_space_max: State

    action_space: gym.spaces.Box
    observation_space: gym.spaces.Box

    _measured_consumption_data: pd.DataFrame
    _measured_pv_production_data: pd.DataFrame
    _measured_wind_production_data: pd.DataFrame
    _spot_market_price_data: pd.DataFrame

    _start_date_data: datetime
    _end_date_data: datetime
    _episode_end_time: datetime

    metadata: Dict[str, List[str]]

    def __init__(
        self,
        data: pd.DataFrame,
        episode_length: timedelta = timedelta(days=30),
        random_seed: Optional[int] = None,
        charge_loss_battery: float = 0.85,
        charge_loss_hydrogen: float = 0.325,
        grid_tariff: float = 0.05,
        peak_grid_tariff: float = 49.0,
    ) -> None:
        """Initialise the environment.

        Args:
            data: Historical data of consumption, wind production, solar (PV)
                production, spot market price data, etc.
            episode_length: The length of an episode.
            random_seed: Optional random seed. It may be used for reproducibility.
            charge_loss_battery: The charging loss in the battery storage.
            charge_loss_hydrogen: The charging loss in the hydrogen storage.
            grid_tariff: The cost of energy imported from the grid.
            peak_grid_tariff: The cost of the peak power imported from the grid.
        """

        """
            data：消耗、风力生产、太阳能（PV）生产、现货市场价格数据等的历史数据。
            episode_length：一个情节的长度。
            random_seed：可选的随机种子，可以用于重现结果。
            charge_loss_battery：电池储能的充电损失。
            charge_loss_hydrogen：氢储能的充电损失。
            grid_tariff：从电网导入能源的成本。
            peak_grid_tariff：从电网导入峰值电力的成本

        """


        # Init random seed if desired, e.g. for reproducibility.
        self.seed(random_seed)

        # Metadata used in render-function (requirement by OpenAiGym):
        self.metadata = {"render.modes": ["ansi"]}

        # The length of the episode and resolutions:
        self._episode_length = episode_length
        self._time_resolution = timedelta(hours=1)  # hourly resolution

        # Loss constants:
        self._charge_loss_battery_storage = charge_loss_battery
        self._charge_loss_hydrogen_storage = charge_loss_hydrogen

        # Reward function constants:
        # Currently we assume winter prices from peak priced grid
        # https://ts.tensio.no/kunde/nettleie-priser-og-avtaler
        self._grid_tariff = grid_tariff  # kr/KWh
        self._peak_grid_tariff = peak_grid_tariff  # kr/kW/mnd

        # Data:
        self._measured_consumption_data = data.consumption
        self._measured_wind_production_data = data.wind_production
        self._measured_pv_production_data = data.pv_production

        self._spot_market_price_data = data.spot_market_price

        # Actions space:
        self._action_space_min = Action(
            charge_battery=-400,  # Geometrical of max
            charge_hydrogen=-100,  # Fuel cell capacity
        )

        self._action_space_max = Action(
            charge_battery=400,  # Maximum charging is 400 KV, 400 KWh/h
            charge_hydrogen=55,  # Electrolyzer has maximum power of 55 KW
        )

        self.action_space = gym.spaces.Box(
            low=self._action_space_min.vector,
            high=self._action_space_max.vector,
            dtype=np.float64,
        )

        # State space:
        self._state_space_min = State(
            consumption=self._measured_consumption_data.min(),
            # Minimum consumption is 0, anything else is measurement noise
            wind_production=self._measured_wind_production_data.min(),
            # Minimum production is 0, anything else is measurement noise
            pv_production=self._measured_pv_production_data.min(),
            # Minimum production is 0, anything else is measurement noise
            spot_market_price=self._spot_market_price_data.min(),
            battery_storage=0,
            hydrogen_storage=0,
            grid_import=0,
            grid_import_peak=0,
            # negative prices in 2020
        )

        self._state_space_max = State(
            consumption=self._measured_consumption_data.max(),
            wind_production=self._measured_wind_production_data.max(),
            pv_production=self._measured_pv_production_data.max(),
            spot_market_price=self._spot_market_price_data.max(),
            battery_storage=500,  # Capacity of battery is 500 Kwh
            hydrogen_storage=1670,  # The tank can hold 100 kg H2, energy density )
            # 33kWh/kg, 50% efficiency
            grid_import=np.inf,
            grid_import_peak=np.inf,
        )

        # Define observation space (here observation space = state space):
        self.observation_space = gym.spaces.Box(
            low=self._state_space_min.vector,
            high=self._state_space_max.vector,
            dtype=np.float64,
        )

        # The start date of possible simulations:
        self._start_time_data = data.index.min()

        # The end date of possible simulations:
        self._end_time_data = data.index.max()

        self.reset()

    def get_possible_start_times(self) -> List[datetime]:
        """Return a list of possible ``start_times`` based on data."""
        index = self._measured_pv_production_data.loc[
            : self._end_time_data - self._episode_length
        ].index
        return list(index)

    def get_state_vector(self) -> np.ndarray:
        """ Returns state vector. """
        return self._state.vector

    def seed(self, random_seed: Optional[int] = None) -> None:
        """Set the seed for this environment's random number generator."""
        if random_seed is not None:
            seed(random_seed)

    def reset(
        self,
        start_time: Optional[datetime] = None,
        battery_storage: float = 0.0,
        hydrogen_storage: float = 0.0,
        grid_import: float = 0.0,
    ) -> np.ndarray:
        """Reset the environment to its initial state and return an initial observation.

        Args:
            start_time: Optional start time of episode.
            battery_storage: Start level of battery_storage.
            hydrogen_storage: Start level of hydrogen storage.
            grid_import: Start level of grid_import.

        Returns:
            state (object): the initial state.
        """


        """
        参数：

            start_time：情节的可选开始时间。
            battery_storage：电池储能的初始水平。
            hydrogen_storage：氢储能的初始水平。
            grid_import：电网输入的初始水平。
        返回值：

            state：初始状态
        """
        # Cumulative _reward for an episode
        self._cumulative_reward = 0

        # Set time attributes
        if start_time is None:
            delta = (self._end_time_data - self._episode_length) - self._start_time_data
            random_seconds = randrange(delta.days * 24 * 60 * 60 + delta.seconds)
            random_hours = int(random_seconds / 3600)
            self._time = self._start_time_data + timedelta(hours=random_hours)

        else:
            self._time = _get_hour_resolution(start_time)

        self._episode_end_time = self._time + self._episode_length

        # Set initial state
        state = State(
            consumption=self._measured_consumption_data.loc[self._time],
            wind_production=self._measured_wind_production_data.loc[self._time],
            pv_production=self._measured_pv_production_data.loc[self._time],
            spot_market_price=self._spot_market_price_data.loc[self._time],
            battery_storage=battery_storage,
            hydrogen_storage=hydrogen_storage,
            grid_import=grid_import,
            grid_import_peak=grid_import,  # NB: Peak is set to last known grid import
        )
        # Make sure initial values are within desired state space
        state_vector = np.clip(
            state.vector,
            a_min=self.observation_space.low,
            a_max=self.observation_space.high,
        )
        self._state = State.from_vector(cast(np.ndarray, state_vector))

        return self._state.vector

    def _perform_action_on_env(
        self,
        action_array: np.ndarray,
        state_current: State,
    ) -> Tuple[State, Action]:
        """Calculate the new states by performing an action on the environment.

        Args:
            action: The action used to generate the new state.

        Returns:
            state_new: The new state, based on current state and action.
            action: The action actually performed in the env.
        """

        """
        参数：

            action：用于生成新状态的动作。
        返回值：

            state_new：基于当前状态和动作的新状态。
            action：在环境中实际执行的动作。
        """

        # Saturate vector
        saturated_action_vector = np.clip(
            action_array,
            a_min=self._action_space_min.vector,
            a_max=self._action_space_max.vector,
        )

        action = Action.from_vector(cast(np.ndarray, saturated_action_vector))

        # Get data for the current timestep
        consumption_new = self._measured_consumption_data.loc[self._time]
        wind_production_new = self._measured_wind_production_data.loc[self._time]
        pv_production_new = self._measured_pv_production_data.loc[self._time]
        spot_market_price = self._spot_market_price_data.loc[self._time]

        # Inflict charging losses from electrical to chemical energy
        if action.charge_battery > 0:
            charge_battery = self._charge_loss_battery_storage * action.charge_battery
        else:
            charge_battery = action.charge_battery

        if action.charge_hydrogen > 0:
            charge_hydrogen = (
                self._charge_loss_hydrogen_storage * action.charge_hydrogen
            )
        else:
            charge_hydrogen = action.charge_hydrogen

        # Constrain battery storage
        battery_storage_new = np.clip(
            state_current.battery_storage + charge_battery,
            a_min=self._state_space_min.battery_storage,
            a_max=self._state_space_max.battery_storage,
        )

        # Constrain hydrogen storage
        hydrogen_storage_new = np.clip(
            state_current.hydrogen_storage + charge_hydrogen,
            a_min=self._state_space_min.hydrogen_storage,
            a_max=self._state_space_max.hydrogen_storage,
        )

        # Need to ensure we can't discharge more than capabilities of battery:
        if action.charge_hydrogen < 0:
            discharge_hydrogen = float(
                hydrogen_storage_new - state_current.hydrogen_storage
            )
            action.charge_hydrogen = max(discharge_hydrogen, action.charge_hydrogen)

        if action.charge_battery < 0:
            discharge_battery = float(
                battery_storage_new - state_current.battery_storage
            )
            action.charge_battery = max(discharge_battery, action.charge_battery)

        # Calculate power which can be used towards consumption in the microgrid
        power_in_microgrid_new = (
            wind_production_new
            + pv_production_new
            - action.charge_hydrogen
            - action.charge_battery
        )

        # Calculate additional power we need from the grid, if negative
        # it means we would originally export power to grid, but this is not allowed
        grid_import_new = np.clip(
            consumption_new - power_in_microgrid_new, a_min=0, a_max=None
        )

        # Calculate new peak
        grid_import_peak_new = max(state_current.grid_import_peak, grid_import_new)

        new_states = State(
            consumption=consumption_new,
            wind_production=wind_production_new,
            pv_production=pv_production_new,
            hydrogen_storage=float(hydrogen_storage_new),
            battery_storage=float(battery_storage_new),
            grid_import=grid_import_new,
            spot_market_price=spot_market_price,
            grid_import_peak=grid_import_peak_new,
        )

        # Return updated states and actions
        return new_states, action

    def _reward(self, state: State, done: bool) -> float:
        """Return reward of being in a state.

        Args:
            state: A state.

        Returns:
            reward: Reward of being in the state.
        """

        power = (state.spot_market_price + self._grid_tariff) * state.grid_import

        if done:
            peak = self._peak_grid_tariff * state.grid_import_peak
        else:
            peak = 0
        peak = 0
        #print(f"Power: {power}, Peak: {peak}")
        reward = power + peak

        return -reward

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Run one-time step of the environment's dynamics.

        When the end of the episode is reached, the environments states are reset.

        Args:
            action (object): An action provided by the agent.

        Returns:
            observation (object): Agent's observation of the current environment.
            reward (float): The amount of _reward returned after the previous action.
            done (bool): Whether the episode has ended, in which case further step()
                calls will return undefined results.
            info (dict): Contains auxiliary state information.
        """

        self._time += self._time_resolution

        new_state, new_action = self._perform_action_on_env(
            action_array=action, state_current=self._state
        )

        # Check if finished episode
        if self._time >= self._episode_end_time:
            done = True
        else:
            done = False

        # Calculate reward
        reward = self._reward(new_state, done)
        self._cumulative_reward += reward

        # Update variables for next iteration
        self._state = new_state

        # Create info
        info = {
            "state": new_state,
            "action": new_action,
            "time": self._time,
            "reward": reward,
            "cumulative_reward": self._cumulative_reward,
        }

        if done:
            self.reset()

        return new_state.vector, reward, done, info

    def render(self, mode: str = "ansi") -> str:
        """Render the environment.

        Args:
            mode (str): The mode to render with.

        Returns:
            Supported mode: ansi:
                Return a string (str) containing a terminal-style text representation.
                The text can include newlines and ANSI escape sequences
                 (e.g. for colours).
        """

        if mode == "ansi":
            return (
                f"Step {self._time}/{self._episode_end_time}"
                f"have state {self._state}"
            )
        else:
            raise Exception(f"Not available mode {mode}")

class RyeFlexEnvFixed(RyeFlexEnv):
    def __init__(self, data):
        super().__init__(data)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space.shape[-1],), dtype=np.float32)

    def step(self, action):
        obs, rewards, done, info = super().step(action)
        return obs.astype(np.float32), rewards.astype(np.float32), done, info

    def reset(self):
        obs = super().reset()
        return obs.astype(np.float32)
    

class RandomActionAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self) -> np.ndarray:
        """
        Normally one would take state as input, and select action based on this.
        Since we are taking random action here, knowing the stat is not necessary.
        """

        return self.action_space.sample()

class SimpleStateBasedAgent:
    """
    An agent which always returns a constant action
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, state) -> np.ndarray:
        """
        Normally one would take the state as input, and select action based on this.
        Since we are taking random action here, knowing the stat is not necessary.
        """

        # Create a state for total production:
        state = State.from_vector(state)
        total_production = state.pv_production + state.wind_production

        if total_production > 20:
            return np.array([0.1, 0])
        else:
            # Charging battery with 0 kWh/h and hydrogen with 10 kWh/h
            return np.array([0, 0.1])
        




@dataclass
class RyeFlexEnvEpisodePlotter:
    """A tool for plotting the states, actions and rewards for an episode.

    Args:
        states: List of states from one episode.
        actions: List of actions from one episode.
        times: List of time-steps from one episode.
        rewards: List of rewards from one episode (both reward and cumulative reward).
    """

    _states: List[Dict[str, float]]
    _actions: List[Dict[str, float]]
    _times: List[datetime]
    _rewards: List[Dict[str, float]]

    def __init__(self) -> None:
        self.reset()

    def update(self, info: Dict[str, Union[State, Action, float, datetime]]) -> None:
        """Update list of states, actions, times and rewards.

        Args:
            info: Info dictionary from the output from env.step(action).
        """
        self._states.append(info["state"].__dict__)
        self._actions.append(info["action"].__dict__)
        self._times.append(cast(datetime, info["time"]))
        self._rewards.append(
            {
                "reward": cast(float, info["reward"]),
                "cumulative_reward": cast(float, info["cumulative_reward"]),
            }
        )

    def plot_episode(self, show: bool = True) -> None:
        """Plot states, rewards and actions from the episode, and there prepare for the
        next episode (reset).

        Args:
            show: Boolean if the plot should be shown.
        """
        _states = pd.DataFrame(self._states, index=self._times)
        _actions = pd.DataFrame(self._actions, index=self._times)
        _reward = pd.DataFrame(self._rewards, index=self._times)

        _states.plot(subplots=True, title="States")

        _actions.plot(subplots=True, title="Actions")

        _reward.plot(subplots=True, title="Rewards")

        if show:
            plt.show()
        self.reset()

    def reset(self) -> None:
        """Reset the list of states, actions, times and rewards."""
        self._states = []
        self._actions = []
        self._times = []
        self._rewards = []