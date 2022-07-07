from doom_enviroment import *
from doom_network_dqn import *
from torch import optim
from doom_agent_dqn import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
case = 'training'

scenario = 'basic'
episodes = 200
update = 5
starting_eps = 0.9
ending_eps = 0.05
total_steps = 7000
batch_size = 256
capacity = 10000
gamma = 0.95
lr = 1e-3
frames = 4
update_target = 5
update_policy = 1
update_eval = 10
update_save = 50


if case == 'training':
    doom_env = VizDoomEnv(render=True)
    policy = DoomDQN(doom_env.get_num_actions(),device).to(device)
    target = DoomDQN(doom_env.get_num_actions(),device).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.RMSprop(policy.parameters(),lr=lr)
    doom_agent = DoomAgentDQN(starting_eps=starting_eps, ending_eps=ending_eps, env=doom_env, policy=policy, target=target,
                              total_steps=total_steps, batch_size= batch_size,capacity=capacity,device=device,gamma=gamma,
                              optimizer=optimizer,update_target=update_target,episodes=episodes,frames=frames,
                              update_policy=update_policy,update_eval=update_eval,update_save=update_save)
    rewards = doom_agent.learn()
    torch.save(doom_agent.policy.state_dict(), "./checkpoint/policy_dqn_{}_{}_{}_{}_ft.pth".format(lr,episodes,batch_size,scenario))
    plt.plot(range(update_eval,episodes+1,update_eval),rewards)
    plt.title("Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig("Reward_basic.png")
    plt.show()
else:
    doom_env = VizDoomEnv(render=True)
    policy = DoomDQN(doom_env.get_num_actions(), device).to(device)
    policy.load_state_dict(torch.load("./final_models/basic.pth"))
    target = DoomDQN(doom_env.get_num_actions(), device).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.RMSprop(policy.parameters(), lr=lr)
    doom_agent = DoomAgentDQN(starting_eps=starting_eps, ending_eps=ending_eps, env=doom_env, policy=policy,
                              target=target,
                              total_steps=total_steps, batch_size=batch_size, capacity=capacity, device=device,
                              gamma=gamma,
                              optimizer=optimizer, update_target=update_target, episodes=episodes, frames=frames,
                              update_policy=update_policy, update_eval=update_eval, update_save=update_save)
    reward = doom_agent.eval()
    print(reward)


