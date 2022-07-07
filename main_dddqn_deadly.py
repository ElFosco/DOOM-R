import numpy as np
import torch

from doom_agent_dddqn import DoomAgentDDQN
from doom_enviroment import *
from doom_network_ddqn import DoomDDQN
from doom_network_dqn import *
from torch import optim
from doom_agent_dqn import *
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

case = 'testing' #training or testing

scenario = "deadly_corridor"
episodes = 5000
difficulty = 3
update_target = 5
starting_eps = 0.9
ending_eps = 0.05
total_steps = 500000
batch_size = 256
capacity = 70000
gamma = 0.95
lr = 5e-4
frames = 4
update_eval = 50
update_save = 1000
update_policy = 10


if case == 'training':
    doom_env = VizDoomEnv(scenario=scenario,difficulty=difficulty,render=True)
    policy = DoomDDQN(doom_env.get_num_actions(),device).to(device)
    target = DoomDDQN(doom_env.get_num_actions(),device).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.RMSprop(policy.parameters(),lr=lr)
    doom_agent = DoomAgentDDQN(starting_eps=starting_eps, ending_eps=ending_eps, env=doom_env, policy=policy, target=target,
                              total_steps=total_steps, batch_size= batch_size,capacity=capacity,device=device,gamma=gamma,
                              optimizer=optimizer,update_target=update_target,update_policy=update_policy,episodes=5000,frames=frames,
                              update_eval=update_eval,update_save=update_save)
    reward_array = doom_agent.learn()
    print(reward_array)
    plt.plot(range(update_eval,episodes+1,update_eval),reward_array)
    plt.title("Evolution of the reward")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig("Reward_deadly.png")
    plt.show()
    torch.save(doom_agent.policy.state_dict(), "./checkpoint/policy_dqn_{}_{}.pth".format(difficulty,scenario))
else:
    doom_env = VizDoomEnv(scenario=scenario, difficulty=difficulty, render=True)
    policy = DoomDDQN(doom_env.get_num_actions(), device).to(device)
    policy.load_state_dict(torch.load("./final_models/policy_dddqn_1000_deadly_3.pth"))
    target = DoomDDQN(doom_env.get_num_actions(), device).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.RMSprop(policy.parameters(), lr=lr)
    doom_agent = DoomAgentDDQN(starting_eps=starting_eps, ending_eps=ending_eps, env=doom_env, policy=policy,
                               target=target,
                               total_steps=total_steps, batch_size=batch_size, capacity=capacity, device=device,
                               gamma=gamma,
                               optimizer=optimizer, update_target=update_target, update_policy=update_policy,
                               episodes=5000, frames=frames,
                               update_eval=update_eval, update_save=update_save)
    reward = doom_agent.eval()
    print(reward)
