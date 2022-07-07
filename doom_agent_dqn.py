import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from skimage import transform
from replay_memory import ReplayMemory
import matplotlib.pyplot as plt

class DoomAgentDQN:
    def __init__(self,starting_eps,ending_eps,env,policy,target,total_steps,batch_size,capacity,device,gamma,
                 optimizer,update_target,episodes,frames,update_policy,update_eval,update_save):
        self.loss = nn.MSELoss()
        self.policy = policy
        self.target = target
        self.starting_eps = starting_eps
        self.ending_eps = ending_eps
        self.total_steps = total_steps
        self.env = env
        self.batch_size = batch_size
        self.memory = ReplayMemory(capacity)
        self.device = device
        self.steps = 0
        self.gamma = gamma
        self.optimizer = optimizer
        self.update_target = update_target
        self.episodes = episodes
        self.frames = frames
        self.update_policy = update_policy
        self.update_eval = update_eval
        self.update_save = update_save

    def preprocessing_img(self,image):
        if self.env.scenario == "basic":
            crop_img = image[100:220,100:400]
            #img = np.squeeze(crop_img)
            #plt.imshow(img)
            #plt.show()
            norm_img = crop_img / 255
            preprocessed_img = transform.resize(norm_img, [84, 84])
            #plt.imshow(preprocessed_img)
            #plt.show()
        if self.env.scenario == "deadly_corridor":
            crop_img = image[100:300,75:425]
            #img = np.squeeze(crop_img)
            #plt.imshow(img)
            #plt.show()
            norm_img = crop_img / 255
            preprocessed_img = transform.resize(norm_img, [84, 84])
            #plt.imshow(preprocessed_img)
            #plt.show()
        if self.env.scenario == "defend_the_center":
            crop_img = image[100:300]
            #img = np.squeeze(crop_img)
            #plt.imshow(img)
            #plt.show()
            norm_img = crop_img / 255
            preprocessed_img = transform.resize(norm_img, [84, 84])
            #plt.imshow(preprocessed_img)
            #plt.show()
        return preprocessed_img

    def stack_starting_img(self,img):
        preprocessed = [img for _ in range(self.frames)]
        return torch.tensor(preprocessed,dtype=torch.float)

    def stack_images(self,stacked_img,state_to_append):
        state_to_append = torch.tensor(state_to_append,dtype=torch.float)
        state_to_append = state_to_append.unsqueeze(0)
        next_state = torch.cat((stacked_img[1:],state_to_append))
        return next_state

    def pick_action(self,state):
        p = random.uniform(0, 1)
        eps = max(self.ending_eps,self.starting_eps - (self.starting_eps - self.ending_eps)*(self.steps/self.total_steps))
        self.steps +=1
        if p < eps:
            action = torch.tensor(np.random.randint(self.env.num_actions),device=self.device, dtype=torch.int32)
        else:
            with torch.no_grad():
                action = torch.argmax(self.policy(state))
        return action

    def update_weight(self):
        total_loss = 0
        self.optimizer.zero_grad()
        samples = self.memory.sample(self.batch_size)
        for sample in samples:
            state_value = self.policy(sample[0])[sample[1]]
            if sample[3]!=None:
                next_state_value = torch.max(self.target(sample[3]))
            else:
                next_state_value = torch.tensor(0,device=self.device)
            expected_action_value = sample[2] + (next_state_value * self.gamma)
            total_loss += self.loss(state_value, expected_action_value)
        total_loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()



    def train_step(self):
        final_reward = 0
        state = self.env.start()
        preprocessed_img = self.preprocessing_img(state)
        stacked_img = self.stack_starting_img(preprocessed_img)
        while True:
            action = self.pick_action(stacked_img)
            state_to_append,reward,done = self.env.step(action.cpu().detach().numpy())
            final_reward +=reward
            if not done:
                state_to_append = self.preprocessing_img(state_to_append)
                next_stacked_img = self.stack_images(stacked_img,state_to_append)
            else:
                next_stacked_img = None
            element = [stacked_img,action,reward,next_stacked_img]
            stacked_img = next_stacked_img
            self.memory.push(element)
            if self.env.scenario == "basic" or self.env.scenario == "deadly_corridor":
                if self.memory.len() >= self.batch_size:
                    if self.steps % self.update_policy == 0:
                        self.update_weight()
            if done:
                if self.env.scenario == "defend_the_center":
                    if self.memory.len() >= self.batch_size:
                        self.update_weight()
                break

    def eval(self):
        if self.env.scenario == "basic":
            reward_list = []
            self.policy.eval()
            for _ in tqdm(range(40)):
                final_reward = 0
                state = self.env.start()
                preprocessed_img = self.preprocessing_img(state)
                stacked_img = self.stack_starting_img(preprocessed_img)
                while True:
                    action = torch.argmax(self.policy(stacked_img))
                    state_to_append, reward, done = self.env.step(action.cpu().detach().numpy())
                    final_reward += reward
                    if done:
                        break
                    else:
                        state_to_append = self.preprocessing_img(state_to_append)
                        stacked_img = self.stack_images(stacked_img, state_to_append)
                reward_list.append(final_reward)
        if self.env.scenario == "defend_the_center":
            reward_list = []
            self.policy.eval()
            for _ in tqdm(range(40)):
                final_reward = 0
                state = self.env.start()
                preprocessed_img = self.preprocessing_img(state)
                stacked_img = self.stack_starting_img(preprocessed_img)
                while True:
                    action = torch.argmax(self.policy(stacked_img))
                    state_to_append, reward, done = self.env.step(action.cpu().detach().numpy())
                    final_reward += 1
                    if done:
                        break
                    else:
                        state_to_append = self.preprocessing_img(state_to_append)
                        stacked_img = self.stack_images(stacked_img, state_to_append)
                reward_list.append(final_reward)
        if self.env.scenario == "deadly_corridor":
            reward_list = []
            self.policy.eval()
            for _ in tqdm(range(5)):
                final_reward = 0
                state = self.env.start()
                preprocessed_img = self.preprocessing_img(state)
                stacked_img = self.stack_starting_img(preprocessed_img)
                while True:
                    action = torch.argmax(self.policy(stacked_img))
                    state_to_append, reward, done = self.env.step(action.cpu().detach().numpy())
                    final_reward += reward
                    if done:
                        break
                    else:
                        state_to_append = self.preprocessing_img(state_to_append)
                        stacked_img = self.stack_images(stacked_img, state_to_append)
                reward_list.append(final_reward)
        return np.mean(np.array(reward_list))

    def learn(self):
        reward_list = []
        for episode in tqdm(range(self.episodes)):
            self.train_step()
            if (episode + 1) % self.update_target == 0:
                self.target.load_state_dict(self.policy.state_dict())
            if (episode + 1) % self.update_eval == 0:
                reward = self.eval()
                reward_list.append(reward)
            if (episode + 1) % self.update_save == 0:
                torch.save(self.policy.state_dict(), "./checkpoint/policy_dqn_{}.pth".format(episode + 1))
        return reward_list
