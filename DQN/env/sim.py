import gym

class env_gym(object):
    def __init__(self, game_name):
        self.env = gym.make(game_name)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def space_size(self, space):
        if space == 'input':
            space_ = self.env.observation_space
        else:
            space_ = self.env.action_space
        try:
            size = space_.n
        except:
            size = space_.shape[0]
        return size

    def close(self):
        return self.env.close()

    def render(self, mode, transpose):
        return self.env.render(mode=mode).transpose(transpose)

    def x_threshold(self):
        return self.env.x_threshold

    def state(self):
        return self.env.state

