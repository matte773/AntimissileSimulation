import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import random
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Define the missile environment
class MissileEnv:    
    def __init__(self, start_position=None, goal_position=None, velocity=100, weight=0.7):
        self.velocity = velocity  # Speed in units per second
        self.bounds = (1000, 1000, 1000) 
        self.start = np.array(start_position, dtype=np.float64) if start_position is not None else np.array([0.0, 0.0, 0.0])
        self.goal = np.array(goal_position, dtype=np.float64) if goal_position is not None else np.array([random.uniform(800, 1000), 
                               random.uniform(0, self.bounds[1]), 
                               random.uniform(0, self.bounds[2])])
        self.position = np.copy(self.start).astype(np.float64)
        self.velocity_vector = np.array([0, 0, 0])  
        self.reached_goal = False
        self.weight = weight  # Weight for evasion strategy

    def reset(self):
        self.position = np.copy(self.start)
        self.path = [self.position]
        self.reached_goal = False

    def step(self, action):
        self.velocity_vector = 0.8 * self.velocity_vector + 0.2 * action
        self.position += self.velocity_vector * self.velocity * 0.2  # Adjust step size

        self.path.append(self.position.copy())

        if np.linalg.norm(self.position - self.goal) < 10:
            self.reached_goal = True

    def get_state(self):
        return self.position.copy()
    
    def is_goal_reached(self):
        return self.reached_goal

    def get_possible_actions(self):
        # Define possible movement directions in 3D
        return [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1]),
            np.array([1, 1, 0]), np.array([-1, -1, 0]),
            np.array([1, 0, 1]), np.array([-1, 0, -1]),
            np.array([0, 1, 1]), np.array([0, -1, -1])
        ]

# Define the POMDP Planner
class POMDPPlanner:
    def __init__(self, env, antimissile, belief_particles=100):
        self.env = env
        self.antimissile = antimissile
        self.belief_particles = belief_particles
        self.belief = [env.get_state() + np.random.normal(0, 2, 3) for _ in range(belief_particles)]  
        # self.belief = [env.get_state() for _ in range(belief_particles)]  

    def update_belief(self, observation):
        self.belief = [obs + np.random.normal(0, 3, 3) for obs in self.belief]  # Example update rule
        # self.belief = [obs for obs in self.belief]  # Example update rule

    def closer_to_goal(self, state, next_state):
        return np.linalg.norm(next_state - self.env.goal) < np.linalg.norm(state - self.env.goal)
    
    def closer_to_antimissile(self, state, next_state):
        return np.linalg.norm(next_state - self.antimissile.get_state()) < np.linalg.norm(state - self.antimissile.get_state())

    def reward(self, state, action, next_state, goal):
        # Compute raw distance changes
        goal_distance_change = np.linalg.norm(state - goal) - np.linalg.norm(next_state - goal)
        antimissile_distance_change = np.linalg.norm(next_state - self.antimissile.get_state()) - np.linalg.norm(state - self.antimissile.get_state())

        # Normalize changes to [0, 1]
        max_goal_change = np.linalg.norm(state - goal)  # Max possible change for goal
        max_antimissile_change = np.linalg.norm(state - self.antimissile.get_state())  # Max possible change for antimissile

        goal_distance_change_norm = goal_distance_change / max_goal_change if max_goal_change != 0 else 0
        antimissile_distance_change_norm = antimissile_distance_change / max_antimissile_change if max_antimissile_change != 0 else 0

        # Weighted reward components
        reward = self.env.weight * goal_distance_change_norm + (1.0 - self.env.weight) * antimissile_distance_change_norm

        return reward
    
    # Most-Likely heuristic: Chooses the action that maximizes immediate reward 
    # based on the most likely state estimate.
    def most_likely_heuristic(self):
        best_action = None
        best_reward = float('-inf')

        for action in self.env.get_possible_actions():
            next_state = self.env.get_state() + action * self.env.velocity #* 0.05  # Simulate next state
            action_reward = self.reward(self.env.get_state(), action, next_state, self.env.goal)

            # Optionally, print the action and its reward
            #print(f"Evaluating action {action}, Reward: {action_reward}")  # Debugging output

            if action_reward > best_reward:
                best_reward = action_reward
                best_action = action

        # Optionally, print the best action and reward
        # print(f"Selected action {best_action} with reward {best_reward}")
        
        return best_action

class AntiMissile:
    def __init__(self, target, initial_position=None, velocity=105, navigation_gain=3.0):
        self.velocity = velocity
        self.bounds = (1000, 1000, 1000)  
        self.start = np.array(initial_position, dtype=np.float64) if initial_position is not None else np.array([random.uniform(800, 1000), random.uniform(0, self.bounds[1]), random.uniform(0, self.bounds[2])])        
        self.position = np.copy(self.start).astype(np.float64)
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

        # Predict future missile position
        predicted_missile_pos = missile_pos + self.target.velocity_vector * 0.2  

        # Adjust movement toward predicted position
        new_direction = predicted_missile_pos - self.position
        new_direction /= np.linalg.norm(new_direction)  

        # Move the anti-missile
        step_distance = new_direction * self.velocity * 0.2  
        self.position += step_distance
        # self.path.append(self.position.copy())

    def get_state(self):
        return self.position.copy()
    
def run_simulation(show_plots, num_simulations, max_steps, missile_position, antimissile_position, goal_position, missile_velocity, antimissile_velocity, weighting_factor):
    missile_reached_goal = 0
    missile_intercepted = 0

    print(f"Running Kalman for {num_simulations} simulations")
    if show_plots:
        print("Showing plots")
    print(f"Max Steps before timeout: {max_steps}")
    print(f"Initial Missile Position: {missile_position}")
    print(f"Initial Anti-Missile Position: {antimissile_position}")
    print(f"Goal Position: {goal_position}")
    print(f"Missile Velocity: {missile_velocity}")
    print(f"Anti-Missile Velocity: {antimissile_velocity}")
    print(f"Goal Weighting Factor: {weighting_factor}")
    print(f"Evasion Weighting Factor: {1.0 - weighting_factor}")

    
    for _ in range(num_simulations):
        # env = MissileEnv()
        env = MissileEnv(start_position=missile_position, goal_position=goal_position, velocity=missile_velocity, weight=weighting_factor)
        antimissile = AntiMissile(env, initial_position=antimissile_position, velocity=antimissile_velocity)
        pomdp = POMDPPlanner(env, antimissile)

        env.reset()
        missile_traj = []
        antimissile_traj = []

        # Simulation update function
        simulation_ended = False
        steps = 0

        start_time = time.perf_counter()

        # Run simulation
        if not show_plots:
            while not simulation_ended:
                if not env.is_goal_reached() and not antimissile.intercepted:
                    observation = env.get_state() + np.random.normal(0, 2, 3)  # Simulated noisy observation
                    pomdp.update_belief(observation)
                    
                    best_action = pomdp.most_likely_heuristic()
                    env.step(best_action)
                    missile_traj.append(env.get_state())

                    antimissile.proportional_navigation()
                    antimissile_traj.append(antimissile.get_state())

                    # Compute distances
                    missile_to_goal_dist = np.linalg.norm(env.get_state() - env.goal)
                    missile_to_antimissile_dist = np.linalg.norm(env.get_state() - antimissile.get_state())

                    # Optional: Print distances at each step
                    # print(f"Missile-Goal Distance = {missile_to_goal_dist:.2f}, "
                    #       f"Missile-AntiMissile Distance = {missile_to_antimissile_dist:.2f}")
                    
                    if steps > max_steps:
                        print("\n--- Simulation Ended ---")
                        print("❌ Simulation ended due to max steps reached")
                        break
                    steps += 1

                else:
                    missile_to_goal_dist = np.linalg.norm(env.get_state() - env.goal)
                    missile_to_antimissile_dist = np.linalg.norm(env.get_state() - antimissile.get_state())

                    # print("\n--- Simulation Ended ---")
                    if env.is_goal_reached():
                        print("✅ Missile reached the goal!")
                        missile_reached_goal += 1
                    elif antimissile.intercepted:
                        print("❌ Anti-Missile intercepted the missile!")
                        missile_intercepted += 1
                        

                    # Optional: Print final distances
                    # print(f"Final Missile-Goal Distance: {missile_to_goal_dist:.2f}")
                    # print(f"Final Missile-AntiMissile Distance: {missile_to_antimissile_dist:.2f}")
                    # print(f"{missile_to_goal_dist:.2f}")
                    # print(f"{missile_to_antimissile_dist:.2f}")

                    # Optional: Print elapsed time
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    # print(f"Elapsed time: {elapsed_time:.4f} seconds")
                    # print(f"{elapsed_time:.4f}")

                    simulation_ended = True
        # Show plots
        else:
            confounding = False
            def update(frame):
                nonlocal simulation_ended, missile_reached_goal, missile_intercepted, confounding, steps

                simulation_ended = False

                if not simulation_ended:
                    if not env.is_goal_reached() and not antimissile.intercepted:
                        observation = env.get_state() + np.random.normal(0, 2, 3)  # Simulated noisy observation
                        pomdp.update_belief(observation)
                        
                        best_action = pomdp.most_likely_heuristic()
                        env.step(best_action)
                        missile_traj.append(env.get_state())

                        antimissile.proportional_navigation()
                        antimissile_traj.append(antimissile.get_state())

                        # Compute distances
                        missile_to_goal_dist = np.linalg.norm(env.get_state() - env.goal)
                        missile_to_antimissile_dist = np.linalg.norm(env.get_state() - antimissile.get_state())

                        # print(f"Step {steps}")

                        # Optional: Print distances at each step
                        # print(f"Frame {frame}: Missile-Goal Distance = {missile_to_goal_dist:.2f}, "
                        #     f"Missile-AntiMissile Distance = {missile_to_antimissile_dist:.2f}")

                        if steps > max_steps:
                            print("\n--- Simulation Ended ---")
                            print("❌ Simulation ended due to max steps reached")
                            confounding = True
                            return
                        steps += 1

                    else:
                        missile_to_goal_dist = np.linalg.norm(env.get_state() - env.goal)
                        missile_to_antimissile_dist = np.linalg.norm(env.get_state() - antimissile.get_state())

                        print("\n--- Simulation Ended ---")
                        if env.is_goal_reached():
                            print("✅ Missile reached the goal!")
                            missile_reached_goal += 1
                            #Optional: uncomment the line below to keep the plots open at the end of the simulation
                            # input("Press Enter to continue...")
                            plt.close()
                        elif antimissile.intercepted:
                            print("❌ Anti-Missile intercepted the missile!")
                            missile_intercepted += 1
                            #Optional: uncomment the line below to keep the plots open at the end of the simulation
                            # input("Press Enter to continue...")
                            plt.close()

                        # Optional: Print final distances
                        # print(f"Final Missile-Goal Distance: {missile_to_goal_dist:.2f}")
                        # print(f"Final Missile-AntiMissile Distance: {missile_to_antimissile_dist:.2f}")


                        if env.is_goal_reached() and antimissile.intercepted:
                            confounding=True

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

                if antimissile.intercepted:
                    ax.scatter(*antimissile.intercept_point, color='green', marker='X', s=200, label='Collision')

                if env.is_goal_reached():
                    ax.scatter(*env.goal, color='red', marker='X', s=200, label='Goal Reached')

                ax.legend()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            def animate():
                ani = FuncAnimation(fig, update, frames=100, interval=200, blit=False)
                # Save before showing the animation
                if confounding:
                    print("Saving Both animation...")
                    ani.save('pomdp_both.gif', writer='pillow', fps=10)
                elif steps > max_steps:
                    print("Saving No Collision animation...")
                    ani.save('pomdp_no_collision.gif', writer='pillow', fps=10)
                # Uncomment to save the animation
                # else:
                #     print("Saving animation...")
                #     ani.save('pomdp.gif', writer='pillow', fps=10)
                plt.show()

            animate()

    return missile_reached_goal, missile_intercepted, num_simulations - missile_reached_goal - missile_intercepted

def main(show_plots=False, num_simulations=100, max_steps=500, missile_position=None, antimissile_position=None, goal_position=None, missile_velocity=100, antimissile_velocity=105, weighting_factor=0.8):
    
    missile_reached_goal, missile_intercepted, max_steps_reached = run_simulation(
        show_plots=show_plots, 
        num_simulations=num_simulations, 
        max_steps=max_steps, 
        missile_position=missile_position, 
        antimissile_position=antimissile_position, 
        goal_position=goal_position, 
        missile_velocity=missile_velocity, 
        antimissile_velocity=antimissile_velocity, 
        weighting_factor=weighting_factor
    )

    print(f"\nSimulation Results for POMDP planner:")
    print(f"Missile reached goal in {missile_reached_goal} out of 100 simulations")
    print(f"Missile intercepted in {missile_intercepted} out of 100 simulations")
    print(f"Number of simulations that reached max steps: {num_simulations - missile_reached_goal - missile_intercepted}")

    print(f"Missile Success Rate: {round(missile_reached_goal/num_simulations *100)}% and Anti-Missile Success Rate: {round((num_simulations - missile_reached_goal)/num_simulations * 100)}%\n")

if __name__ == "__main__":
    main()