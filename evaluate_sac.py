# evaluate_sac.py
import gymnasium as gym
import gym_pybullet_drones
from stable_baselines3 import SAC
import numpy as np
import time

def evaluate_sac():
    print("TESTING SAC MODEL")
    print("=" * 40)
    
    try:
        model = SAC.load("models/sac/sac_model")
    except:
        print("SAC model not found. Train first: python train_sac.py")
        return
    
    env = gym.make("hover-aviary-v0", gui=True)
    
    print("Starting SAC visualization...")
    print("Close PyBullet window to stop")
    
    rewards = []
    
    for episode in range(3):
        obs, info = env.reset()
        done = truncated = False
        episode_reward = 0
        steps = 0
        
        while not done and not truncated and steps < 1000:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            time.sleep(0.01)
            
            if done or truncated:
                break
        
        rewards.append(episode_reward)
        print(f"Episode {episode+1}: {steps} steps, Reward = {episode_reward:.2f}")
    
    print(f"\nSAC Average Reward: {np.mean(rewards):.2f}")
    env.close()

if __name__ == "__main__":
    evaluate_sac()