#!/usr/bin/python

import random
import numpy as np
import math

from carla.client import CarlaClient
from carla.settings import CarlaSettings
from gym.spaces import Box

from carla import image_converter
from carla.sensor import Camera

npc_vehicle_seeds  = [248879577, 284212177, 249989477, 464819363, 196289952, 859931552, 873340571, 113509344, 856798697, 422790201, 119129994]
start_list_indices = [18, 28, 91, 91, 136, 98, 68, 24, 51, 121, 141]
weathers           = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

class gym_carla_car_following:
    def __init__(self, host="127.0.0.1", port=2000, timeout=15):

        # Steer, Throttle/Brake
        self.action_space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))

        self.speed_scale      = 50       # km/h
        self.relative_x_scale = 4.0      # m
        self.relative_y_scale = 40.0     # m
        self.relative_angle_scale = 180  # degree

        self._history_path_num = 60

        low = np.zeros([self._history_path_num * 2 + 5])
        high = np.ones([self._history_path_num * 2 + 5])

        # Only shape of low/high array matters
        self.observation_space = Box(low=low, high=high)

        self._host = host
        self._port = port
        self._timeout = timeout

    def reset(self):

        # connect to server
        self._client = CarlaClient(self._host, self._port, self._timeout)
        self._client.connect()

        # settings
        # index = random.randint(0, len(npc_vehicle_seeds) - 2) # Last one is for testing
        # print(index)
        index = 0
        seed = npc_vehicle_seeds[index]
        start_list_index = start_list_indices[index]

        settings = CarlaSettings()
        settings.set(SynchronousMode=True,
                     SendNonPlayerAgentsInfo=True,
                     NumberOfVehicles=1,
                     NumberOfPedestrians=0,
                     SeedVehicles=seed,
                     SeedPedestrians=123456789,
                     WeatherId=weathers[index])

        self._client.load_settings(settings)
        self._client.start_episode(start_list_index)

        # simulator init stage
        for i in range(10):
            measurements, _ = self._client.read_data()
            self._client.send_control(steer=0, throttle=0, brake=0, hand_brake=False, reverse=False)

        self._client.send_control(steer=0, throttle=0, brake=0, hand_brake=False, reverse=False)

        # Reset Flags
        self._reset = True
        self._history_path = np.zeros([self._history_path_num, 4])
        self._cur_steer = 0
        self._his_steer = 0

        observation = self._observe()

        self._observation = observation

        return observation

    def step(self, action):

        steering     = action[0]

        # filter small value
        if action[1] >= 0.01:
            acceleration = action[1]
            brake = 0
        elif action[1] <= -0.01:
            acceleration = 0
            brake = -action[1]
        else:
            acceleration = 0
            brake = 0

        # one command for every 2 frame
        for i in range(2):
            self._client.send_control(steer=steering, throttle=acceleration, brake=brake, hand_brake=False, reverse=False)

        observation = self._observe()

        info = {}
        done   = self._calculate_done(observation[:5])
        reward = self._calculate_reward(observation[:5], done)

        self._his_steer = self._cur_steer
        self._cur_steer = steering

        return observation, reward, done, info

    def stop(self):
        self._client.disconnect()

        # Wait for server to be connectable again
        import time
        time.sleep(8)

    def print_state(self, step, state, action, reward, epsilon=0):
        print("Step %d : epsilong=%f" % (step, epsilon))
        print("    State : self_speed=%f(km/h), npc_speed=%f(km/h), rel_angle=%f(degree), "
              "rel_x=%f(m), rel_y=%f(m), last_rel_x=%f(m), last_rel_y=%f(m)"
              % (state[0] * self.speed_scale, state[1] * self.speed_scale, state[2] * self.relative_angle_scale,
                 state[3] * self.relative_x_scale, state[4] * self.relative_y_scale,
                 state[-2] * self.relative_x_scale, state[-1] * self.relative_y_scale))
        print("    Action : steer=%f, throttle/brake=%f, reward=%f" % (action[0], action[1], reward))

    def _observe(self):

        #################################### current state #######################################
        measurements, _ = self._client.read_data()
        self_car   = measurements.player_measurements
        self_speed = self_car.forward_speed
        npc_car  = measurements.non_player_agents[-1].vehicle
        npc_speed = npc_car.forward_speed

        # normalization
        self_speed /= self.speed_scale
        if self_speed < 0.0:
            self_speed = 0.0
        npc_speed /= self.speed_scale
        if npc_speed < 0.0:
            npc_speed = 0.0

        # Coordination in this simulator is like this :
        #         ^ +
        #         I
        #         I
        # <-------0--------
        # +       I       -
        #         I -

        # cm -> m
        npc_pos_x = - npc_car.transform.location.x / 100.0
        npc_pos_y = npc_car.transform.location.y / 100.0
        npc_orientation_x = - npc_car.transform.orientation.x
        npc_orientation_y = npc_car.transform.orientation.y

        pos_x = - self_car.transform.location.x / 100.0
        pos_y = self_car.transform.location.y / 100.0
        orientation_x = - self_car.transform.orientation.x
        orientation_y = self_car.transform.orientation.y

        [npc_relative_x, npc_relative_y, npc_relative_angle] = \
            self._transform_coordination(pos_x, pos_y, orientation_x, orientation_y,
                                         npc_pos_x, npc_pos_y, npc_orientation_x, npc_orientation_y)

        # normalization
        npc_relative_x /= self.relative_x_scale
        npc_relative_y /= self.relative_y_scale
        npc_relative_angle /= self.relative_angle_scale
        npc_relative_angle = np.clip(npc_relative_angle, -1.0, 1.0)

        observation = np.zeros([self._history_path_num * 2 + 5])
        observation[0] = self_speed
        observation[1] = npc_speed
        observation[2] = npc_relative_angle
        observation[3] = npc_relative_x
        observation[4] = npc_relative_y

        #################################### history state #######################################
        state = np.hstack((npc_pos_x, npc_pos_y, npc_orientation_x, npc_orientation_y))
        if self._reset == True:
            self._history_path[:] = state
            self._reset = False
            for i in range(self._history_path_num):
                observation[5 + (self._history_path_num - 1 - i) * 2] = npc_relative_x
                observation[5 + (self._history_path_num - 1 - i) * 2 + 1] = npc_relative_y
        else:
            last_his_num = 0
            for i in range(self._history_path_num):
                npc_pos_x = self._history_path[i, 0]
                npc_pos_y = self._history_path[i, 1]
                npc_orientation_x = self._history_path[i, 2]
                npc_orientation_y = self._history_path[i, 3]
                [npc_relative_x, npc_relative_y, npc_relative_angle] = \
                    self._transform_coordination(pos_x, pos_y, orientation_x, orientation_y,
                            npc_pos_x, npc_pos_y, npc_orientation_x, npc_orientation_y)

                # normalization
                npc_relative_x /= self.relative_x_scale
                npc_relative_y /= self.relative_y_scale
                npc_relative_angle /= self.relative_angle_scale
                npc_relative_angle = np.clip(npc_relative_angle, -1.0, 1.0)
                if npc_relative_y > 0.025:
                    observation[5 + i * 2] = npc_relative_x
                    observation[5 + i * 2 + 1] = npc_relative_y
                else:
                    last_his_num = i
                    # print("********************************************************")
                    # print(observation[5 + i * 2 - 2], observation[5 + i * 2 + 1 - 2])
                    # print("********************************************************")

                    break


            if last_his_num > 0:
                # print("**********************************************")
                # print(last_his_num)
                # print("*************************************************")
                for i in range(last_his_num, self._history_path_num):
                    observation[5 + i * 2] = observation[5 + (i - 1) * 2]
                    observation[5 + i * 2 + 1] = observation[5 + (i - 1) * 2 + 1]

            self._history_path = np.roll(self._history_path, 1, axis=0)
            self._history_path[0] = state
        return observation

    def _calculate_reward(self, observation, done):

        speed     = observation[0]
        npc_speed = observation[1]
        rel_angle = observation[2]
        rel_x     = observation[3]
        rel_y     = observation[4]

        # reward = 0
        if done == True:
            reward = -1000.0
        else:
            # How to calculate reward ?
            reward = ((rel_y - 0.5)**2) * 20.0 \
                        - abs(rel_x) * 20.0 + \
                        - abs(rel_angle - 1.0 / 2) * 20 \
                        - abs(speed - npc_speed) * 20.0 \
                        - abs(self._cur_steer - self._his_steer) * 20

        return reward

    def _calculate_done(self, observation):

        speed     = observation[0]
        npc_speed = observation[1]
        rel_angle = observation[2]
        rel_x     = observation[3]
        rel_y     = observation[4]

        done = False
        # if (abs(rel_x) >= 2.5) or \
                # (rel_y >= 2.5) or (rel_y <= 0.15) or \
                # (rel_angle < 1.0 / 8) or (rel_angle > 7.0 / 8):
            # done = True
        distance = math.sqrt((rel_x * self.relative_x_scale)**2 + \
                       (rel_y * self.relative_y_scale)**2)

        if (distance > 100) or \
                (rel_angle < -0.01):
            done = True
        return done

    def _normalize_angle(self, angle):
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def _transform_coordination(self, pos_x, pos_y, orientation_x, orientation_y,
                                npc_pos_x, npc_pos_y, npc_orientation_x, npc_orientation_y):

        absolute_angle     = math.atan2(orientation_y, orientation_x) * 180 / 3.1415926
        npc_absolute_angle = math.atan2(npc_orientation_y, npc_orientation_x) * 180 / 3.1415926

        # print("self %f %f with %f %f %f" % (pos_x, pos_y, orientation_x, orientation_y, absolute_angle))
        # print("npc %f %f with %f %f %f" % (npc_pos_x, npc_pos_y, npc_orientation_x, npc_orientation_y, npc_absolute_angle))

        delta_pos_x = npc_pos_x - pos_x
        delta_pos_y = npc_pos_y - pos_y

        npc_absolute_pos_angle = math.atan2(delta_pos_y, delta_pos_x) * 180 / 3.1415926

        # print("delta_x = %f, delta_y = %f " % (delta_pos_x, delta_pos_y))

        npc_relative_angle = (90 - absolute_angle ) + npc_absolute_angle
        npc_relative_pos_angle = (90 - absolute_angle) + npc_absolute_pos_angle

        distance = math.sqrt(delta_pos_x ** 2 + delta_pos_y ** 2)

        npc_relative_pos_x = distance * math.cos(npc_relative_pos_angle * 3.1415926 / 180)
        npc_relative_pos_y = distance * math.sin(npc_relative_pos_angle * 3.1415926 / 180)

        npc_relative_angle = self._normalize_angle(npc_relative_angle)

        # print("relative %f %f %f %f" % (npc_relative_pos_x, npc_relative_pos_y, npc_relative_angle, distance))

        return [npc_relative_pos_x, npc_relative_pos_y, npc_relative_angle]


if __name__ == "__main__":
    env = None
    try:
        env   = gym_carla_car_following("127.0.0.1", 2000, 15)
        state = env.reset()
        i = 0
        while True:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            env.print_state(i, state, action, reward)
            state = next_state
            i += 1
            if done == True:
                break
    finally:
        if env is not None:
            env.stop()
