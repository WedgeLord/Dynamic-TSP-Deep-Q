import csv
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

import matplotlib.pyplot as plt
import imageio

# cleans up output
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class DTSPEnv(Env):
    def __init__(self, world_size=10, position=(0, 0), max_cities=3, episode_length=100, cities=(()), name='test', visualize=False):
        self.world_size = world_size
        self.start = list(position)
        self.max_cities = max_cities
        self.episode_length = episode_length
        self.cities = list(cities)
        # deadline scales with problem size
        self.max_deadline = world_size * max_cities
        # visualization parameters
        self.frames = []
        self.last_position = self.start
        self.name = name
        self.visualize = visualize

        # inherited Environment attributes
        # self.action_space = Discrete(max_cities, start=0)  # max queue size = max_cities
        self.action_space = Discrete(math.factorial(max_cities), start=0)  # !max_cities possible combinations
        # travel queue, agent position, remaining nodes
        self.observation_space = Dict({
            "time": Box(0, episode_length, dtype=np.int32),
            "position": Box(low=np.array([0, 0]), high=np.array([world_size, world_size]), shape=(2,), dtype=np.int32),
            "city_position": Box(low=0, high=world_size, shape=(max_cities, 2), dtype=np.int32),
            # fixme: idk the right 'high' value
            "deadline": Box(low=0, high=self.episode_length+self.max_cities*self.max_deadline, shape=(max_cities,), dtype=np.int32),
            "new_city": Dict({
                "pos": Box(low=0, high=world_size, shape=(2,), dtype=np.int32),
                "ded": Discrete(self.max_deadline + 1)  # +1 so it includes bounds
            })
        })
        # self.reward_range = (-np.inf, 0)  # default value is (-inf, inf)

        # mutable state variables
        self.state = dict({
            "time": 0,
            "position": position,
            "city_position": [[],],
            # "city_position": [[-1, -1] for _ in range(self.max_cities)],
            "deadline": [],
            # "deadline": [-1 for _ in range(self.max_cities)],
            "new_city": dict({
                # "pos": [-1, -1],
                # "ded": -1,
                "pos": [],
                "ded": -1,
            }),
        })
        self.reset()

    # render doesn't really fit with this model's implementation, use snapshot() instead
    def render(self, mode):
        pass

    def setVis(self, vis):
        self.visualize = vis
    def snapshot(self, action):
        if self.visualize is False:
            return
        fig, ax = plt.subplots(figsize=(self.world_size, self.world_size))
        plt.title(f'time: {self.state["time"]} || Action: {action}')
        # plot movement vector
        p_ = self.last_position
        p0 = self.state["position"]
        V = [p0[0] - p_[0], p0[1] - p_[1]]
        ax.arrow(p_[0], p_[1], V[0], V[1], head_width=0.5, head_length=0.5)
        self.last_position = [*p0]
        ax.scatter(p0[0], p0[1], color='black', s=500)
        path = self.state["city_position"]
        # plot cities
        for n in range(self.max_cities):
            p = path[n]
            if p == [-1, -1]:
                break
            ax.scatter(p[0], p[1], color='green', s=300)
            ded = self.state["deadline"][n] - self.state["time"]  # relative deadline
            if ded < self.world_size:
                color = 'red'
            else:
                color = 'black'
            ax.annotate("Ded_{}".format(ded), xy=(p[0] + 0.4, p[1] + 0.05), zorder=2, c=color)
        # plot new city node
        new = self.state["new_city"]["pos"]
        if new != [-1, -1]:
            color = 'red' if action == 0 else 'yellow'
            ax.scatter(new[0], new[1], color=color, s=300)
            # resets new_city, remove this line to keep the new_city node off-colored in future snapshots
            self.state["new_city"]["pos"] = [-1, -1]
        path = [self.state["position"], *path]
        color = 'blue' if action % self.action_space.n == 0 else 'red'
        # draws updated path
        for i in range(self.max_cities):
            p1 = path[i]
            p2 = path[i+1]
            if p2 == [-1, -1]:
                break
            ax.plot([p1[0], p2[0]],
                    [p1[1], p2[1]], linestyle='solid', color=color)
        ax.set_xlim(-1, self.world_size+1)
        ax.set_ylim(-1, self.world_size+1)
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return image

    def close(self):
        plt.close()

    # resets data, adding one random city to queue
    def reset(self):
        self.state["time"] = 0
        self.state["position"] = self.start
        self.state["city_position"] = [[-1, -1] for _ in range(self.max_cities)]
        self.state["deadline"] = [-1 for _ in range(self.max_cities)]
        self.state["new_city"]["pos"] = [r.randint(0, self.world_size) for _ in range(2)]
        self.state["new_city"]["ded"] = r.randint(self.world_size, self.max_deadline)
        if len(self.cities) > 0:
            r.shuffle(self.cities)
        self.last_position = self.start
        self.steps = 0
        self.frames = []
        return self.state

    def move(self):
        if self.state["city_position"][0] == [-1, -1]:
            return 0
        dif = [0, 0]
        for i in range(2):
            dif[i] = self.state["city_position"][0][i] - self.state["position"][i]
        abs_dif = [abs(dif[0]), abs(dif[1])]
        if abs_dif[0] > 0.5:
            self.state["position"][0] += dif[0] / abs_dif[0]
        elif abs_dif[1] > 0.5:
            self.state["position"][1] += dif[1] / abs_dif[1]
        else:
            # agent is already at city (within 1x1 square), remove city and move all other cities up the queue
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
        self.state["deadline"][action] = self.state["new_city"]["ded"] + self.state["time"]
        # below commented for rendering purposes
        # self.state["new_city"]["pos"] = [-1, -1]
        # self.state["new_city"]["ded"] = -1
        return 1

    def reroute(self, action):
        if action % math.factorial(self.max_cities) == 0:  # no-change
            return
        route = [[-1, -1] for _ in range(self.max_cities)]
        deds = [-1 for _ in range(self.max_cities)]
        switch = [c for c in range(self.max_cities+1)]  # +1 allows us to use 1-based indexing
        home = [c for c in range(self.max_cities+1)]
        city = 2  # start checking switches at second city
        while(action != 0):
            if self.state["deadline"][city - 1] == -1:
                break
            target = action % city
            if target != 0:  # city marked for switch
                switch[city] = target
            action //= city
            city += 1
        for c in reversed(range(1, self.max_cities+1)):
            # route is 0-based index
            route[home[switch[c]] - 1] = self.state["city_position"][c - 1]
            deds[home[switch[c]] - 1] = self.state["deadline"][c - 1]
            home[switch[c]] = home[c]
        self.state["city_position"] = route
        self.state["deadline"] = deds

    def step(self, action):
        reward = self.add_city(action)
        self.reroute(action)
        # add new city every few time units
        requested = False
        while not requested:
            if self.state["time"] < self.episode_length:
                # todo: adjust this request-timing distribution to your needs
                requested = r.randint(0, 4) == 0
            elif self.state["city_position"][0] == [-1, -1]:
                # episode is finished and no more cities left
                break
            # the line below stores the frame for gif creation
            self.frames.append(self.snapshot(action))
            # the line below stores the snapshot in a folder (folder must exist)
            # imageio.imwrite(f'./solutions/{self.name}/{self.state["time"]}.png', self.snapshot(action))
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
            done = False
            self.state["new_city"]["ded"] = r.randint(self.world_size, self.max_deadline)
            if len(self.cities) > 0:
                if len(self.cities) < self.steps:  # no more cities to visit
                    done = True
                else:
                    self.state["new_city"]["pos"] = self.cities[self.steps]
            else:
                self.state["new_city"]["pos"] = [r.randint(0, self.world_size) for _ in range(2)]
        else:  # no more new cities, we can calculate completed route plan now
            done = True
        if done:
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
        self.steps += 1
        info = {}  # required for some reason
        return self.state, reward, done, info


def build_model(states, actions):
    model = Sequential()
    # '*' notation unpacks data, the 1, is required for some reason
    model.add(Flatten(input_shape=(1, *states)))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=0.01)
    return dqn


def makeGif(agent, env, title):
    # stores value for reset later
    visualize = env.visualize
    env.reset()
    env.setVis(True)
    agent.test(env, nb_episodes=1, visualize=False)
    env.setVis(False)
    kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
    imageio.mimsave(f'./solutions/test{title}.gif', env.frames, fps=2)


if __name__ == '__main__':
    data = []
    with open('data.txt') as datafile:
        reader = csv.reader(datafile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        for line in reader:
            line[:] = [num for num in line if num]  # removes empty strings from line
            data.append(line)
    max = data[0][0]
    for city in data:
        if max < city[0]:
            max = city[0]
        if max < city[1]:
            max = city[1]
    for city in data:
        city[:] = [x * 10 / max for x in city]  # scales down to world size
    print(data)
    # Dict-type Spaces cannot be used for testing, they must be flattened first
    env = FlattenObservation(DTSPEnv())
    test_env = FlattenObservation(DTSPEnv(cities=data, visualize=True))
    # build model
    states = env.observation_space.shape
    actions = env.action_space.n
    model = build_model(states, actions)

    # create agent
    # the same agent can be used in different environments if their models are identical
    deepQ = build_agent(model, actions)
    deepQ.compile(Adam(lr=0.005), metrics=[keras.metrics.RootMeanSquaredError()])
    # deepQ.compile(Adam(lr=0.001), metrics=['mae'])

    # get some visuals
    makeGif(deepQ, env, 'random')
    makeGif(deepQ, test_env, 'untrained')
    deepQ.test(env, nb_episodes=20, visualize=False)  # initial test
    deepQ.test(test_env, visualize=False)
    deepQ.fit(env, nb_steps=50000, visualize=False, verbose=1)
    makeGif(deepQ, test_env, 'trained')
    for i in range(3):
        makeGif(deepQ, env, i)
    env.reset()
    # deepQ.test(env, nb_episodes=20, visualize=False)
