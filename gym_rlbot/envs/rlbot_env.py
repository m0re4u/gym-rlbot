import math

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
import gym
from gym import spaces
from rlbot import runner
from rlbot.setup_manager import SetupManager

import multiprocessing as mp
import numpy as np

class RLBotEnv(gym.Env):
    def __init__(self):
        super(RLBotEnv, self).__init__()
        self.action_space = spaces.Tuple((
            spaces.Box(low=np.array([-1., -1., -1., -1., -1.]), high=np.array([1., 1., 1., 1., 1.]), dtype=np.float), 
            spaces.Discrete(2),
            spaces.Discrete(2),
            spaces.Discrete(2)
            ))
        self.observation_space = spaces.Box(low=np.array([-1.,-1]), high=np.array([1,1]), dtype=np.float)
        manager = runner.main(gym=True)        
        p = mp.Process(target=manager.infinite_loop)
        print(manager.infinite_loop)
        p.start()


    def step(self, action):
        self.agent.act(action)
        obs = self.agent._get_obs()
        reward = 0
        scored = False

        return obs, reward, scored, {}

    def reset(self):
        return 
  
    def render(self, mode='human', close=False):
        pass


class Vector2:
    def __init__(self, x=0, y=0):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, val):
        return Vector2(self.x + val.x, self.y + val.y)

    def __sub__(self, val):
        return Vector2(self.x - val.x, self.y - val.y)

    def correction_to(self, ideal):
        # The in-game axes are left handed, so use -x
        current_in_radians = math.atan2(self.y, -self.x)
        ideal_in_radians = math.atan2(ideal.y, -ideal.x)

        correction = ideal_in_radians - current_in_radians

        # Make sure we go the 'short way'
        if abs(correction) > math.pi:
            if correction < 0:
                correction += 2 * math.pi
            else:
                correction -= 2 * math.pi

        return correction


class GymAgent(BaseAgent):
    def initialize_agent(self):
        #This runs once before the bot starts up
        self.controller_state = SimpleControllerState()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        ball_location = Vector2(packet.game_ball.physics.location.x, packet.game_ball.physics.location.y)

        self.obs = [packet.game_ball.physics.location.x, packet.game_ball.physics.location.y]
        
        
        my_car = packet.game_cars[self.index]
        self.controller_state.throttle = 0.0
        self.controller_state.steer = 0.0

        draw_debug(self.renderer, my_car, packet.game_ball, "test")

        return self.controller_state

    def _get_obs(self):
        return self.obs

    def act(self, action):
        self.controller_state.throttle = action[0][0]
        self.controller_state.steer = action[0][1]
        self.controller_state.pitch = action[0][2]
        self.controller_state.yaw = action[0][3]
        self.controller_state.roll = action[0][4]
        self.controller_state.jump =action[1]
        self.controller_state.boost = action[2]
        self.controller_state.handbrake = action[3]

 
def get_car_facing_vector(car):
    pitch = float(car.physics.rotation.pitch)
    yaw = float(car.physics.rotation.yaw)

    facing_x = math.cos(pitch) * math.cos(yaw)
    facing_y = math.cos(pitch) * math.sin(yaw)

    return Vector2(facing_x, facing_y)

def draw_debug(renderer, car, ball, action_display):
    renderer.begin_rendering()
    # draw a line from the car to the ball
    renderer.draw_line_3d(car.physics.location, ball.physics.location, renderer.white())
    # print the action that the bot is taking
    renderer.draw_string_3d(car.physics.location, 2, 2, action_display, renderer.white())
    renderer.end_rendering()




# # 'physics': {
# #     'location': {'x': 0.0, 'y': 0.0, 'z': 0.0},
# #     'rotation': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
# #     'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
# #     'angular_velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0}
# # },
# self_location = spaces.Box(low=np.array([-1., -1., -1]), high=np.array([1., 1., 1]), dtype=np.float)
# self_rotation = spaces.Box(low=np.array([-1., -1., -1]), high=np.array([1., 1., 1]), dtype=np.float)
# self_velocity = spaces.Box(low=np.array([-1., -1., -1]), high=np.array([1., 1., 1]), dtype=np.float)
# self_angular_velocity = spaces.Box(low=np.array([-1., -1., -1]), high=np.array([1., 1., 1]), dtype=np.float)
# has_wheel_contact = spaces.Discrete(2)
# jumped = spaces.Discrete(2)
# boost = spaces.Box(low=np.array([0]), high=np.array([100]), dtype=np.float)
# ball_location = spaces.Box(low=np.array([-1., -1., -1]), high=np.array([1., 1., 1]), dtype=np.float)
# ball_rotation = spaces.Box(low=np.array([-1., -1., -1]), high=np.array([1., 1., 1]), dtype=np.float)
# ball_velocity = spaces.Box(low=np.array([-1., -1., -1]), high=np.array([1., 1., 1]), dtype=np.float)
# ball_angular_velocity = spaces.Box(low=np.array([-1., -1., -1]), high=np.array([1., 1., 1]), dtype=np.float)

# self.observation_space = spaces.Tuple((
#   self_location, 
#   self_rotation, 
#   self_velocity, 
#   self_angular_velocity, 
#   has_wheel_contact,
#   jumped,
#   boost,
#   ball_location,
#   ball_rotation,
#   ball_velocity,
#   ball_angular_velocity,
#   ))