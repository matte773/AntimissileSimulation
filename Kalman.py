import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import random

# Simulation Constants
number_of_simulations = 1000
max_steps = 10000
show_plots = False

missile_reached_goal = 0
missile_intercepted = 0


# Define the missile environment
class MissileEnv:
    def __init__(self, bounds=(1000, 1000, 1000), velocity=100):
        self.bounds = bounds
        self.velocity = velocity  # Speed in units per second
        self.start = np.array([0.0, 0.0, 0.0])
        self.goal = np.array([random.uniform(800, 1000), 
                               random.uniform(0, bounds[1]), 
                               random.uniform(0, bounds[2])])
        self.position = np.copy(self.start)
        self.velocity_vector = np.array([0, 0, 0])  
        self.reached_goal = False  
        
    def reset(self):
        self.position = np.copy(self.start)
        self.reached_goal = False

    def step(self, action):
        self.velocity_vector = 0.9 * self.velocity_vector + 0.3 * action  # Increase momentum influence
        self.position += self.velocity_vector * self.velocity * 0.08  # Increase step size
  
        if np.linalg.norm(self.position - self.goal) < 10:
            self.reached_goal = True

    def get_state(self):
        return self.position.copy()
    
    def is_goal_reached(self):
        return self.reached_goal

# Kalman Filter for tracking missile movement
class KalmanFilter:
    def __init__(self, initial_state, process_noise=1.0, measurement_noise=10.0):
        self.state = np.hstack([initial_state, np.zeros(3)])  # [x, y, z, vx, vy, vz]
        self.P = np.eye(6) * 100  
        self.Q = np.eye(6) * 5.0  # Increase process noise
        self.R = np.eye(3) * measurement_noise  
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])  
        self.I = np.eye(6)  

    def predict(self, dt=0.05):
        F = np.eye(6)
        for i in range(3):
            F[i, i+3] = dt  
        self.state = F @ self.state  
        self.P = F @ self.P @ F.T + self.Q  

    def update(self, measurement):
        z = measurement.reshape(3, 1)  
        y = z - (self.H @ self.state).reshape(3, 1)  
        S = self.H @ self.P @ self.H.T + self.R  
        K = self.P @ self.H.T @ np.linalg.inv(S)  
        self.state = self.state + (K @ y).flatten()  
        self.P = (self.I - K @ self.H) @ self.P  

    def get_position(self):
        return self.state[:3]

    def get_velocity(self):
        return self.state[3:]

# Anti-Missile System (Proportional Navigation)
class AntiMissile:
    def __init__(self, target, bounds=(1000, 1000, 1000), velocity=105, navigation_gain=3.0):
        self.velocity = velocity  
        self.position = np.array([random.uniform(800, 1000), random.uniform(0, bounds[1]), random.uniform(0, bounds[2])])
        self.target = target
        self.intercepted = False
        self.intercept_point = None
        self.navigation_gain = navigation_gain  
        
    def proportional_navigation(self):
        missile_pos = self.target.get_state()
        los_vector = missile_pos - self.position
        los_distance = np.linalg.norm(los_vector)

        if los_distance < 10:
            self.intercepted = True
            self.intercept_point = missile_pos.copy()
            return

        predicted_missile_pos = missile_pos + self.target.velocity_vector * 0.2  
        new_direction = predicted_missile_pos - self.position
        new_direction /= np.linalg.norm(new_direction)  

        step_distance = new_direction * self.velocity * 0.2  
        self.position += step_distance  

    def get_state(self):
        return self.position.copy()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def run_simulation(max_steps=100, show_plots=False, num_simulations=100):
    global missile_reached_goal, missile_intercepted
    
    missile_reached_goal = 0
    missile_intercepted = 0
    
    for _ in range(num_simulations):
        env = MissileEnv()
        antimissile = AntiMissile(env)
        kf = KalmanFilter(env.get_state())
        
        env.reset()
        missile_traj = []
        antimissile_traj = []
        simulation_ended = False
        steps = 0
        
        if not show_plots:
            while not simulation_ended:
                if not env.is_goal_reached() and not antimissile.intercepted:
                    kf.predict(dt=0.05)
                    antimissile.proportional_navigation()
                    
                    noisy_measurement = env.get_state() + np.random.normal(0, 10, 3)
                    kf.update(noisy_measurement)
                    
                    goal_direction = env.goal - kf.get_position()
                    evade_direction = kf.get_position() - antimissile.get_state()
                    combined_direction = goal_direction + 0.5 * evade_direction  
                    combined_direction /= np.linalg.norm(combined_direction)
                    
                    env.step(combined_direction)
                    missile_traj.append(env.get_state())
                    antimissile_traj.append(antimissile.get_state())
                    
                    if steps > max_steps:
                        print("❌ Simulation ended due to max steps reached")
                        break
                    steps += 1
                else:
                    if env.is_goal_reached():
                        print("✅ Missile reached the goal!")
                        missile_reached_goal += 1
                    elif antimissile.intercepted:
                        print("❌ Anti-Missile intercepted the missile!")
                        missile_intercepted += 1
                    simulation_ended = True
        else:
            def update(frame):
                nonlocal simulation_ended
                if not simulation_ended:
                    if not env.is_goal_reached() and not antimissile.intercepted:
                        kf.predict(dt=0.05)
                        antimissile.proportional_navigation()
                        
                        noisy_measurement = env.get_state() + np.random.normal(0, 10, 3)
                        kf.update(noisy_measurement)
                        
                        goal_direction = env.goal - kf.get_position()
                        evade_direction = kf.get_position() - antimissile.get_state()
                        combined_direction = goal_direction + 0.5 * evade_direction  
                        combined_direction /= np.linalg.norm(combined_direction)
                        
                        env.step(combined_direction)
                        missile_traj.append(env.get_state())
                        antimissile_traj.append(antimissile.get_state())
                    else:
                        if env.is_goal_reached():
                            print("✅ Missile reached the goal!")
                            missile_reached_goal += 1
                        elif antimissile.intercepted:
                            print("❌ Anti-Missile intercepted the missile!")
                            missile_intercepted += 1
                        plt.close()
                        simulation_ended = True
                
                ax.clear()
                ax.set_xlim(0, env.bounds[0])
                ax.set_ylim(0, env.bounds[1])
                ax.set_zlim(0, env.bounds[2])
                ax.scatter(*env.goal, color='red', s=100, label='Goal')
                ax.scatter(*env.get_state(), color='blue', s=150, label='Missile')
                ax.scatter(*antimissile.get_state(), color='green', s=150, label='Anti-Missile')
                
                if len(missile_traj) > 1:
                    missile_path = np.array(missile_traj)
                    ax.plot(missile_path[:, 0], missile_path[:, 1], missile_path[:, 2], color='blue', linestyle='solid', label='Missile Trajectory')
                
                if len(antimissile_traj) > 1:
                    antimissile_path = np.array(antimissile_traj)
                    ax.plot(antimissile_path[:, 0], antimissile_path[:, 1], antimissile_path[:, 2], color='green', linestyle='solid', label='Anti-Missile Trajectory')
                
                ax.legend()
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ani = FuncAnimation(fig, update, frames=100, interval=200, blit=False)
            plt.show()
    
    return missile_reached_goal, missile_intercepted, num_simulations - missile_reached_goal - missile_intercepted


if __name__ == "__main__":
    missile_reached_goal, missile_intercepted, max_steps_reached = run_simulation(100, show_plots=show_plots, num_simulations=number_of_simulations)
    

    print(f"\nSimulation Results for Kalman Avoidance and Greedy Path planning:")
    print(f"Missile reached goal in {missile_reached_goal} out of {number_of_simulations} simulations")
    print(f"Missile intercepted in {missile_intercepted} out of {number_of_simulations} simulations")
    print(f"Number of simulations that reached max steps: {max_steps_reached}")
    print(f"Missile Success Rate: {round(missile_reached_goal / number_of_simulations * 100)}% and Anti-Missile Success Rate: {round((number_of_simulations - missile_reached_goal) / number_of_simulations * 100)}%\n")