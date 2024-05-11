from gym_battlesnake.gymbattlesnake import BattlesnakeEnv
#from gym_battlesnake.custompolicy import CustomPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO

env = BattlesnakeEnv(n_threads=4, n_envs=16)

model = PPO(ActorCriticPolicy, env, verbose=1, learning_rate=1e-3)
model.learn(total_timesteps=100000)
model.save('ppo2_trainedmodel')

del model

model = PPO.load('ppo2_trainedmodel')

obs = env.reset()
for _ in range(10000):
    action,_ = model.predict(obs)
    obs,_,_,_ = env.step(action)
    env.render()