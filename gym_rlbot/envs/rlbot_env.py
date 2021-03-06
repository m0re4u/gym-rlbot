import math

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
import gym
from gym import spaces
from rlbot import runner
from rlbot.setup_manager import SetupManager

import threading
from multiprocessing import Queue
import numpy as np


class RLBotEnv(gym.Env):
    def __init__(self):
        super(RLBotEnv, self).__init__()
        self.action_space = spaces.Tuple((
            spaces.Box(low=np.array([-1.]), high=np.array([1.]), dtype=np.float),
            # spaces.Box(low=np.array([-1., -1., -1., -1., -1.]), high=np.array([1., 1., 1., 1., 1.]), dtype=np.float),
            # spaces.Discrete(2),
            # spaces.Discrete(2),
            # spaces.Discrete(2)
            ))
        self.manager = runner.main(gym=True)
        self.game_thread = threading.Thread(target=self.manager.infinite_loop)
        self.game_thread.start()

        self.score = 0

        self_location = spaces.Box(low=np.array([-4096., -5120., 0.]),       high=np.array([4096.,5120.,2044.]),      dtype=np.float)
        self_rotation = spaces.Box(low=np.array([-np.pi, -np.pi, -np.pi]), high=np.array([np.pi, np.pi, np.pi]), dtype=np.float)
        # self_velocity = spaces.Box(low=np.array([0., 0., 0.]),               high=np.array([2300.,2300.,2300.]),    dtype=np.float)
        # self_angular_velocity = spaces.Box(low=np.array([-1., -1., -1.]), high=np.array([1., 1., 1.]), dtype=np.float)
        # has_wheel_contact = spaces.Discrete(2)
        # jumped = spaces.Discrete(2)
        # boost = spaces.Box(low=np.array([0]), high=np.array([100]), dtype=np.float)
        ball_location = spaces.Box(low=np.array([-4096., -5120., 0.]),       high=np.array([4096.,5120.,2044.]),      dtype=np.float)
        # ball_rotation = spaces.Box(low=np.array([-np.pi, -np.pi, -np.pi]), high=np.array([np.pi, np.pi, np.pi]), dtype=np.float)
        # ball_velocity = spaces.Box(low=np.array([0., 0., 0.]),               high=np.array([2300.,2300.,2300.]),    dtype=np.float)
        # ball_angular_velocity = spaces.Box(low=np.array([-1., -1., -1.]), high=np.array([1., 1., 1.]), dtype=np.float)

        self.observation_space = spaces.Tuple((
            self_location,
            self_rotation,
            # self_velocity,
            # self_angular_velocity,
            # has_wheel_contact,
            # jumped,
            # boost,
            ball_location,
            # ball_rotation,
            # ball_velocity,
            # ball_angular_velocity,
        ))


    def step(self, action):
        self.act(action)
        obs, scored = self._get_obs()

        if scored:
            reward = 0
        else:
            reward = -1

        return obs, reward, scored, {}

    def reset(self):
        self.manager.reset_game()
        self.act([0]*8)
        obs, _ = self._get_obs()
        return obs

    def render(self, mode='human', close=False):
        pass

    def _get_obs(self):
        packet = self.manager.agent_state_queue.get()
        self_location = [packet.game_cars[0].physics.location.x,  packet.game_cars[0].physics.location.y, packet.game_cars[0].physics.location.z]
        self_rotation = [packet.game_cars[0].physics.rotation.pitch, packet.game_cars[0].physics.rotation.yaw, packet.game_cars[0].physics.rotation.roll]
        self_velocity = [packet.game_cars[0].physics.velocity.x, packet.game_cars[0].physics.velocity.y, packet.game_cars[0].physics.velocity.z]
        ball_location = [packet.game_ball.physics.location.x,  packet.game_ball.physics.location.y, packet.game_ball.physics.location.z]
        ball_rotation = [packet.game_ball.physics.rotation.pitch, packet.game_ball.physics.rotation.yaw, packet.game_ball.physics.rotation.roll]
        ball_velocity = [packet.game_ball.physics.velocity.x, packet.game_ball.physics.velocity.y, packet.game_ball.physics.velocity.z]

        obs = [
            *self_location,
            *self_rotation,
            # *self_velocity,
            # packet.game_cars[0].has_wheel_contact,
            # packet.game_cars[0].jumped,
            # packet.game_cars[0].boost,
            *ball_location,
            # *ball_rotation,
            # *ball_velocity
        ]

        scored = packet.teams[0].score > self.score
        self.score = packet.teams[0].score
        return obs, scored

    def act(self, action):
        controller_state = SimpleControllerState()
        controller_state.throttle = action[0]
        # controller_state.steer = action[1]
        # controller_state.pitch = action[2]
        # controller_state.yaw = action[3]
        # controller_state.roll = action[4]
        # controller_state.jump = True if action[5] > 0.5 else False
        # controller_state.boost = True if action[6] > 0.5 else False
        # controller_state.handbrake = True if action[7] > 0.5 else False
        controller_state.steer = 0
        controller_state.pitch = 0
        controller_state.yaw = 0
        controller_state.roll = 0
        controller_state.jump = 0
        controller_state.boost = 0
        controller_state.handbrake = 0
        self.manager.agent_action_queue.put(controller_state)


class GymAgent(BaseAgent):
    def initialize_agent(self):
        #This runs once before the bot starts up
        self.controller_state = SimpleControllerState()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # All stuff here does not matter anymore, controller state is handled through the gym environment

        my_car = packet.game_cars[self.index]
        draw_debug(self.renderer, my_car, packet.game_ball, "test")

        return None


def draw_debug(renderer, car, ball, action_display):
    renderer.begin_rendering()
    # draw a line from the car to the ball
    renderer.draw_line_3d(car.physics.location, ball.physics.location, renderer.white())
    # print the action that the bot is taking
    renderer.draw_string_3d(car.physics.location, 2, 2, action_display, renderer.white())
    renderer.end_rendering()




