# evaluate_ddpg.py
import gymnasium as gym
import gym_pybullet_drones
from stable_baselines3 import DDPG
import numpy as np
import time

def evaluate_ddpg():
    print("TESTING DDPG MODEL")
    print("=" * 40)
    
    try:
        model = DDPG.load("models/ddpg/ddpg_model")
        print("DDPG model loaded successfully")
    except Exception as e:
        print(f"DDPG model not found: {e}")
        print("Train first: python train_ddpg.py")
        return
    
    env = gym.make("hover-aviary-v0", gui=True)
    
    print("Starting DDPG visualization...")
    print("Close PyBullet window to stop")
    
    rewards = []
    episode_lengths = []
    
    for episode in range(3):
        obs, info = env.reset()
        done = truncated = False
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while not done and not truncated and steps < 1000:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            time.sleep(0.01)
            
            if done or truncated:
                break
        
        rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"   Steps: {steps}, Reward: {episode_reward:.2f}")
    
    print(f"\nDDPG PERFORMANCE:")
    print(f"   Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"   Average Episode Length: {np.mean(episode_lengths):.1f} steps")
    print(f"   Best Reward: {np.max(rewards):.2f}")
    
    env.close()

if __name__ == "__main__":
    evaluate_ddpg()