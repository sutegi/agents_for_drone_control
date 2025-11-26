# train_ppo.py
import gymnasium as gym
import gym_pybullet_drones
from stable_baselines3 import PPO
import os
import time

def train_ppo():
    print("TRAINING PPO MODEL")
    print("=" * 40)
    
    os.makedirs("models/ppo", exist_ok=True)
    
    env = gym.make("hover-aviary-v0", gui=False)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./tensorboard/ppo/"
    )
    
    print("Starting PPO training")
    start_time = time.time()
    
    model.learn(total_timesteps=500000)
    
    training_time = time.time() - start_time
    model.save("models/ppo/ppo_model")
    
    print(f"PPO trained in {training_time:.2f} seconds")
    print("Model saved: models/ppo/ppo_model")
    
    env.close()

if __name__ == "__main__":
    train_ppo()