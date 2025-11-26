# compare_algorithms.py
import gymnasium as gym
import gym_pybullet_drones
from stable_baselines3 import SAC, PPO
import numpy as np

def compare_algorithms():
    print("COMPARING SAC vs PPO")
    print("=" * 40)
    
    env = gym.make("hover-aviary-v0", gui=False)
    
    # Test SAC
    try:
        sac_model = SAC.load("models/sac/sac_model")
        sac_rewards = evaluate_model(sac_model, env, "SAC")
    except:
        print("SAC model not found, training first...")
        return
    
    # Test PPO  
    try:
        ppo_model = PPO.load("models/ppo/ppo_model")
        ppo_rewards = evaluate_model(ppo_model, env, "PPO")
    except:
        print("PPO model not found, training first...")
        return
    
    # Results
    print(f"\nCOMPARISON RESULTS:")
    print(f"SAC:  Avg reward = {np.mean(sac_rewards):.2f}, Std = {np.std(sac_rewards):.2f}")
    print(f"PPO:  Avg reward = {np.mean(ppo_rewards):.2f}, Std = {np.std(ppo_rewards):.2f}")
    
    if np.mean(ppo_rewards) > np.mean(sac_rewards):
        print("PPO performs better than SAC")
    else:
        print("SAC performs better than PPO")
    
    env.close()

def evaluate_model(model, env, name):
    print(f"Evaluating {name}...")
    rewards = []
    
    for episode in range(3):
        obs, info = env.reset()
        done = truncated = False
        episode_reward = 0
        
        while not done and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
        print(f"  {name} Episode {episode+1}: Reward = {episode_reward:.2f}")
    
    return rewards

if __name__ == "__main__":
    compare_algorithms()