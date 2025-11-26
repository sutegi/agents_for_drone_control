# main_project.py
import os

def print_menu():
    print("\n" + "="*50)
    print("PROJECT: SAC vs PPO Drone Comparison")
    print("="*50)
    print("1. Train SAC model")
    print("2. Train PPO model") 
    print("3. Compare algorithms")
    print("4. Test SAC model")
    print("5. Test PPO model")
    print("6. Generate results report")
    print("7. Show project info")
    print("0. Exit")

def main():
    while True:
        print_menu()
        choice = input("\nSelect option (0-7): ").strip()
        
        if choice == "1":
            os.system('python train_sac.py')
        elif choice == "2":
            os.system('python train_ppo.py')
        elif choice == "3":
            os.system('python compare_algorithms.py')
        elif choice == "4":
            os.system('python evaluate_sac.py')
        elif choice == "5":
            os.system('python evaluate_ppo.py')
        elif choice == "6":
            os.system('python generate_report.py')
        elif choice == "7":
            show_project_info()
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice.")
        
        input("\nPress Enter to continue...")

def show_project_info():
    info = """
PROJECT INFO:
Comparison of SAC vs PPO for drone hovering.
We train both algorithms and compare performance.
    
Key Results:
- SAC: 5.52 average reward, 4 steps per episode
- PPO: 44.99 average reward, 160+ steps per episode
    
PPO significantly outperforms SAC for this task.
"""
    print(info)

if __name__ == "__main__":
    main()