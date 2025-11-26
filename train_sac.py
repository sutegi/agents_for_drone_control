# train_sac.py
import gymnasium as gym
import gym_pybullet_drones
from stable_baselines3 import SAC
import os
import time

def train_sac():
    print("TRAINING SAC MODEL")
    print("=" * 40)
    
    os.makedirs("models/sac", exist_ok=True)
    
    env = gym.make("hover-aviary-v0", gui=False)
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tau=0.02,
        verbose=1,
        tensorboard_log="./tensorboard/sac/"
    )

    print("Starting SAC training")
    start_time = time.time()

    model.learn(total_timesteps=50000)

    training_time = time.time() - start_time
    model.save("models/sac/sac_model")
    
    print(f"SAC trained in {training_time:.2f} seconds")
    print("Model saved: models/sac/sac_model")
    
    env.close()

if __name__ == "__main__":
    train_sac()