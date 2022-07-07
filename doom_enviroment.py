import numpy as np
from vizdoom import *
from gym import Env



class VizDoomEnv(Env):

    def __init__(self, scenario="basic",difficulty=5,render=False):
        super().__init__()

        self.scenario = scenario
        self.game = DoomGame()
        self.game.load_config('./scenarios/{}.cfg'.format(scenario))

        self.game.set_doom_skill(difficulty)
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_screen_resolution(ScreenResolution.RES_512X384)
        self.game.init()
        self.num_actions = len(self.game.get_available_buttons())

        self.damage_taken = 0
        self.ammo = 26
        self.hit_count = 0

    def step(self, action):
        actions = np.identity(self.num_actions)
        reward = self.game.make_action(actions[action],4)
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            if self.scenario == "defend_the_center":
                info = self.game.get_state().game_variables
                current_ammo,health = info
                ammo_reward = current_ammo - self.ammo
                self.ammo = current_ammo
                reward = (reward*2 + ammo_reward)
            elif self.scenario == "deadly_corridor":
                info = self.game.get_state().game_variables
                current_damage_taken,current_hit_count,current_ammo = info
                has_lost_health = self.damage_taken - current_damage_taken
                if has_lost_health < 0:
                    damage_taken_reward = -1
                else:
                    damage_taken_reward = 0
                hit_count_reward = current_hit_count - self.hit_count
                ammo_reward = current_ammo - self.ammo

                self.damage_taken = current_damage_taken
                self.hit_count = current_hit_count
                self.ammo = current_ammo
                reward = reward * 200*hit_count_reward + 5*ammo_reward + 10*damage_taken_reward
        else:
            state = None
        done = self.game.is_episode_finished()
        return state, reward, done

    def is_finished(self):
        return self.game.is_episode_finished()

    def start(self):
        self.game.new_episode()
        return self.game.get_state().screen_buffer

    def get_num_actions(self):
        return self.num_actions

