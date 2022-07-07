import numpy as np
import torch

from doom_agent_dddqn import DoomAgentDDQN
from doom_enviroment import *
from doom_network_ddqn import DoomDDQN
from doom_network_dqn import *
from torch import optim
from doom_agent_dqn import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

case = 'testing' #training or testing

scenario = 'defend_the_center'
episodes = 5000
starting_eps = 0.9
ending_eps = 0.05
total_steps = 700000
batch_size = 256
capacity = 70000
gamma = 0.95
lr = 1e-4
frames = 4
update_target = 5
update_policy = 5
update_eval = 50
update_save = 1000


if case == 'training':
    doom_env = VizDoomEnv(render=True,scenario=scenario,difficulty=3)
    policy = DoomDDQN(doom_env.get_num_actions(),device).to(device)
    target = DoomDDQN(doom_env.get_num_actions(),device).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.RMSprop(policy.parameters(),lr=lr)
    doom_agent = DoomAgentDDQN(starting_eps=starting_eps, ending_eps=ending_eps, env=doom_env, policy=policy, target=target,
                              total_steps=total_steps, batch_size= batch_size,capacity=capacity,device=device,gamma=gamma,
                              optimizer=optimizer,update_target=update_target,episodes=episodes,frames=frames,
                              update_policy=update_policy,update_eval=update_eval,update_save=update_save)
    survive_time = doom_agent.learn()
    torch.save(doom_agent.policy.state_dict(), "./checkpoint/policy_dddqn_{}_{}_{}_{}_ft.pth".format(lr,episodes,batch_size,scenario))
    plt.plot(range(update_eval,episodes+1,update_eval),survive_time)
    plt.title("Survive time")
    plt.xlabel("Episodes")
    plt.ylabel("Survive time")
    plt.savefig("Reward_defend_the_center.png")
    plt.show()
else:
    doom_env = VizDoomEnv(render=True, scenario=scenario, difficulty=3)
    policy = DoomDDQN(doom_env.get_num_actions(), device).to(device)
    policy.load_state_dict(torch.load("./final_models/policy_dddqn_defend_the_center.pth"))
    target = DoomDDQN(doom_env.get_num_actions(), device).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.RMSprop(policy.parameters(), lr=lr)
    doom_agent = DoomAgentDDQN(starting_eps=starting_eps, ending_eps=ending_eps, env=doom_env, policy=policy,
                               target=target,
                               total_steps=total_steps, batch_size=batch_size, capacity=capacity, device=device,
                               gamma=gamma,
                               optimizer=optimizer, update_target=update_target, episodes=episodes, frames=frames,
                               update_policy=update_policy, update_eval=update_eval, update_save=update_save)
    survive_time = doom_agent.eval()
    print(survive_time)

