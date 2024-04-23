import math
import random as r

import keras.metrics
import numpy as np
from gym import Env
from gym.spaces import Box, MultiDiscrete, Discrete, Dict
from gym.wrappers import FlattenObservation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# cleans up output
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class DTSPEnv(Env):
    def __init__(self, world_size=10, position=(0, 0), max_cities=3, episode_length=100):
        self.world_size = world_size
        self.max_cities = max_cities
        self.episode_length = episode_length
        self.max_deadline = episode_length + world_size*2 * max_cities

        # inherited Environment attributes
        # self.action_space = Discrete(max_cities, start=0)  # max queue size = max_cities
        self.action_space = Discrete(math.factorial(max_cities)-1, start=0)  # !max_cities possible combinations
        # self.action_space = MultiDiscrete([max_cities for _ in range(max_cities)], dtype=np.int32)
        # travel queue, agent position, remaining nodes
        self.observation_space = Dict({
            "time": Box(0, episode_length, dtype=np.int32),
            "position": Box(low=np.array([0, 0]), high=np.array([world_size, world_size]), shape=(2,), dtype=np.int32),
            "city_position": Box(low=0, high=world_size, shape=(max_cities, 2), dtype=np.int32),
            # fixme: idk the right 'high' value
            "deadline": Box(low=0, high=episode_length+world_size*max_cities, shape=(max_cities,), dtype=np.int32),
            "new_city": Dict({
                "pos": Box(low=0, high=world_size, shape=(2,), dtype=np.int32),
                "ded": Discrete(self.max_deadline)
            })
        })
        # self.reward_range = (-np.inf, 0)  # default value is (-inf, inf)

        # mutable state variables
        self.state = dict({
            "position": position,
            "city_position": [[-1, -1] for _ in range(self.max_cities)],
            "deadline": [-1 for _ in range(self.max_cities)],
            "new_city": dict({
                "pos": [-1, -1],
                "ded": -1,
            }),
            "time": 0,
        })
        self.reset()

    def render(self):
        pass

    # resets data, adding one random city to queue
    def reset(self):
        self.state["time"] = 0
        self.state["position"] = [0, 0]  # depot
        self.state["city_position"] = [[-1, -1] for _ in range(self.max_cities)]
        # self.state["city_position"][0] = [r.randint(0, self.world_size) for _ in range(2)]
        self.state["deadline"] = [-1 for _ in range(self.max_cities)]
        #fixme improve random range
        self.state["new_city"]["pos"] = [r.randint(0, self.world_size) for _ in range(2)]
        self.state["new_city"]["ded"] = r.randint(self.world_size, self.world_size * self.max_cities)
        return self.state

    def move(self):
        if self.state["city_position"][0] == [-1, -1]:
            return 0
        dif = [0, 0]
        for i in range(2):
            dif[i] = self.state["city_position"][0][i] - self.state["position"][i]
        if dif[0] != 0:
            self.state["position"][0] += dif[0] / abs(dif[0])
        elif dif[1] != 0:
            self.state["position"][1] += dif[1] / abs(dif[1])
        else:
            # agent is already at city, remove city and move all other cities up the queue
            for i in range(self.max_cities - 1):
                self.state["city_position"][i] = self.state["city_position"][i + 1]
                self.state["deadline"][i] = self.state["deadline"][i + 1]
            self.state["city_position"][self.max_cities - 1] = [-1, -1]
            self.state["deadline"][self.max_cities - 1] = -1
            # move towards next city in list
            return 1 + self.move()
        return 0
        #     return self.move()
        # return -1

    def add_city(self, action):
        if self.state["city_position"][self.max_cities-1] != [-1, -1]:  # city queue exceeds limit
            self.state["new_city"]["pos"] = [-1, -1]
            self.state["new_city"]["ded"] = -1
            return 0
        if action == 0:  # city not taken
            self.state["new_city"]["pos"] = [-1, -1]
            self.state["new_city"]["ded"] = -1
            return 0
        #new_city = [r.randint(0, self.world_size) for _ in range(2)]
        # places city in queue at earliest available spot, starting at 'action' index
        action = self.max_cities-1  # todo: remove
        while action > 0:
            if self.state["city_position"][action - 1] == [-1, -1]:
                action -= 1
            else:
                break
        # insert city at the agent's choice, moving other cities back in the queue
        # todo uncomment?
        # for city in reversed(range(action, self.max_cities-1)):
        #     self.state["city_position"][city + 1] = self.state["city_position"][city]
        #     self.state["deadline"][city + 1] = self.state["deadline"][city]
        # # finds an open index in position array
        # # self.state["city_position"][action] = new_city
        self.state["city_position"][action] = self.state["new_city"]["pos"]
        self.state["deadline"][action] = self.state["new_city"]["ded"]
        self.state["new_city"]["pos"] = [-1, -1]
        self.state["new_city"]["ded"] = -1
        # self.state["deadline"][action] = self.state["time"] + r.randint(self.world_size, self.world_size * self.max_cities)
        return 1

# todo add "no change" option
    def reroute(self, action):
        route = [[-1, -1] for _ in range(self.max_cities)]
        switch = [c for c in range(self.max_cities+1)]  # +1 allows us to use 1-based indexing
        home = [c for c in range(self.max_cities+1)]
        city = 2
        while(action != 0):
            if self.state["deadline"][city - 1] == -1:
                break
            target = action % city
            if target == 0:  # city isn't marked for switch
                pass
            else:
                switch[city] = target
            action //= city
            city += 1
        # for c in reversed(range(self.max_cities+1)):
        for c in reversed(range(1, self.max_cities+1)):
            # route is 0-based index
            route[home[switch[c]] - 1] = self.state["city_position"][c - 1]
            home[switch[c]] = home[c]
        self.state["city_position"] = route


    def step(self, action):
        reward = self.add_city(action)
        self.reroute(action)
        # add new city every few time units
        for _ in range(5):
            self.move()
            self.state["time"] += 1

            # checks for missed deadlines
            empty = 0
            for d in range(self.max_cities):
                if self.state["deadline"][d] == -1:
                    break
                if self.state["deadline"][d] < self.state["time"]:
                    empty += 1
                    reward -= 1
                elif empty > 0:
                    self.state["city_position"][d - empty] = self.state["city_position"][d]
                    self.state["deadline"][d - empty] = self.state["deadline"][d]
                if empty > 0:
                    self.state["city_position"][d] = [-1, -1]
                    self.state["deadline"][d] = -1

            if self.state["time"] < self.episode_length:
                self.state["new_city"]["pos"] = [r.randint(0, self.world_size) for _ in range(2)]
                self.state["new_city"]["ded"] = self.state["time"] + r.randint(self.world_size * 5, self.world_size * 10)
                # fixme random range
                done = False
            else:  # no more new cities, we can calculate completed route plan now
                for c in range(self.max_cities):
                    if self.state["deadline"][c] == -1:  # no cities left
                        break
                    else:
                        distance = abs(self.state["city_position"][c][0] - self.state["position"][0])
                        distance += abs(self.state["city_position"][c][1] - self.state["position"][1])
                        window = self.state["deadline"][c] - self.state["time"]
                        if distance > window:
                            reward -= 1
                        else:
                            self.state["time"] -= distance
                done = True
        info = {}  # required for some reason
        return self.state, reward, done, info


def build_model(states, actions):
    model = Sequential()
    # '*' notation unpacks data, the 1, is required for some reason
    model.add(Flatten(input_shape=(1, *states)))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=0.01)
    return dqn


if __name__ == '__main__':
    # Dict-type Spaces cannot be used for testing, they must be flattened first
    env = FlattenObservation(DTSPEnv(world_size=15, max_cities=5, episode_length=100))
    states = env.observation_space.shape
    actions = env.action_space.n
    model = build_model(states, actions)

    # score = 0
    # done = False
    # while not done:
    #     env.render()
    #     action = r.choice(range(10))
    #     new_state, reward, done, info = env.step(action)
    #     score += reward
    #     i = new_state[0]
    #     print('pos=({},{}), city=({},{})'.format(new_state[20], new_state[21], new_state[10 + 2 * i],
    #                                              new_state[11 + 2 * i]))
    env.reset()

    deepQ = build_agent(model, actions)
    deepQ.compile(Adam(lr=0.001), metrics=[keras.metrics.RootMeanSquaredError()])
    # deepQ.compile(Adam(lr=0.001), metrics=['mae'])
    deepQ.test(env, nb_episodes=20, visualize=False)  # initial test
    deepQ.fit(env, nb_steps=100000, visualize=False, verbose=1)
    deepQ.test(env, nb_episodes=20, visualize=False)
