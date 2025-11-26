# train_ddpg.py
import gymnasium as gym
import gym_pybullet_drones
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import os
import time

def train_ddpg():
    print("TRAINING DDPG MODEL")
    print("=" * 40)
    
    os.makedirs("models/ddpg", exist_ok=True)
    
    # Create environment
    env = gym.make("hover-aviary-v0", gui=False)
    
    # Define action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    # Create DDPG model
    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=100,
        gamma=0.99,
        tau=0.005,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log="./tensorboard/ddpg/"
    )
    
    print("Starting DDPG training...")
    print(f"Action space: {env.action_space.shape}")
    print(f"Observation space: {env.observation_space.shape}")
    
    start_time = time.time()
    
    # Train the model
    model.learn(total_timesteps=100000)
    
    training_time = time.time() - start_time
    model.save("models/ddpg/ddpg_model")
    
    print(f"DDPG trained in {training_time:.2f} seconds")
    print("Model saved: models/ddpg/ddpg_model")
    
    # Quick test
    test_ddpg_quick(model)
    
    env.close()

def test_ddpg_quick(model):
    """Quick test of the trained DDPG model"""
    print("\nQUICK TEST OF DDPG MODEL")
    
    env = gym.make("hover-aviary-v0", gui=True)
    
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    print("Testing DDPG for 200 steps...")
    
    for step in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if done or truncated:
            break
    
    print(f"DDPG Test: {steps} steps, Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    train_ddpg()