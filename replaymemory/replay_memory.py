from collections import deque
import random

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self,experience):
        self.memory.append(experience)

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    def len(self):
        return len(self.memory)