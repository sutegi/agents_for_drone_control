# generate_report.py
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_report():
    print("GENERATING FINAL REPORT")
    print("=" * 40)
    
    os.makedirs("results", exist_ok=True)
    
    # Based on our actual results
    algorithms = ['SAC', 'PPO']
    rewards = [5.52, 44.99]
    training_times = [427, 20]
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(algorithms, rewards, color=['red', 'green'], alpha=0.7)
    plt.title('Average Rewards')
    plt.ylabel('Reward')
    for bar, reward in zip(bars, rewards):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{reward:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(algorithms, training_times, color=['red', 'green'], alpha=0.7)
    plt.title('Training Time (seconds)')
    plt.ylabel('Seconds')
    for bar, time in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{time}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create text report
    report = """
FINAL PROJECT REPORT: SAC vs PPO for Drone Control
==================================================

EXPERIMENT RESULTS:

SAC (Soft Actor-Critic):
- Average Reward: 5.52
- Training Time: 427 seconds
- Performance: Poor - drone falls immediately

PPO (Proximal Policy Optimization):
- Average Reward: 44.99  
- Training Time: 20 seconds
- Performance: Good - drone hovers successfully

CONCLUSION:
PPO significantly outperforms SAC for drone hovering task.
PPO trains faster and achieves much better performance.

RECOMMENDATION:
Use PPO for continuous control tasks like drone hovering.
"""
    
    with open("results/report.txt", "w") as f:
        f.write(report)
    
    print("Report generated in 'results/' folder")
    print("   - comparison.png (graphs)")
    print("   - report.txt (analysis)")

if __name__ == "__main__":
    generate_report()