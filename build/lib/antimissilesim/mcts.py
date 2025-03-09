import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import random

# Define the missile environment
class MissileEnv: 
    def __init__(self, start_position=None, goal_position=None, velocity=100, weight=0.3):
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
        self.velocity_vector = 0.8 * self.velocity_vector + 0.2 * action  # Add momentum
        self.position += self.velocity_vector * self.velocity * 0.05  # Adjust step size
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

# Monte Carlo Tree Search Node
class MCTSNode:
    def __init__(self, state, antimissile_state, parent=None):
        self.state = state
        self.antimissile_state = antimissile_state  # Track anti-missile position
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def expand(self, env):
        for action in env.get_possible_actions():
            new_state = self.state + action * env.velocity * 0.05  # Reduce movement step
            self.children.append(MCTSNode(new_state, self.antimissile_state, parent=self))

    def best_child(self, exploration_weight=1.0):
        return max(self.children, key=lambda child: (
            child.value / (child.visits + 1e-6) + 
            exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6))
        ))

    def update(self, reward):
        self.visits += 1
        self.value += reward

# MCTS Planning
class MCTSPlanner:
    def __init__(self, env, antimissile, iterations=100):
        self.env = env
        self.antimissile = antimissile
        self.iterations = iterations

    def search(self):
        root = MCTSNode(self.env.get_state(), self.antimissile.get_state())

        for _ in range(self.iterations):
            node = root
            while node.children:
                node = node.best_child()

            if not node.children:
                node.expand(self.env)

            # Reward function balances goal-seeking and evasion
            reward = -np.linalg.norm(node.state - self.env.goal) + \
                    self.env.weight * np.linalg.norm(node.state - self.antimissile.get_state())

            # Backpropagate rewards
            while node:
                node.update(reward)
                node = node.parent

        return root.best_child(exploration_weight=0)

# Define the anti-missile system with proportional navigation
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

    print(f"Running MCTS for {num_simulations} simulations")
    if show_plots:
        print("Showing plots")
    print(f"Max Steps before timeout: {max_steps}")
    print(f"Initial Missile Position: {missile_position}")
    print(f"Initial Anti-Missile Position: {antimissile_position}")
    print(f"Goal Position: {goal_position}")
    print(f"Missile Velocity: {missile_velocity}")
    print(f"Anti-Missile Velocity: {antimissile_velocity}")
    print(f"Evasion Weighting Factor: {weighting_factor}")
    
    for _ in range(num_simulations):
        # env = MissileEnv()
        env = MissileEnv(start_position=missile_position, goal_position=goal_position, velocity=missile_velocity, weight=weighting_factor)
        antimissile = AntiMissile(env, initial_position=antimissile_position, velocity=antimissile_velocity)        
        mcts = MCTSPlanner(env, antimissile, iterations=100)
        
        env.reset()
        missile_traj = []
        antimissile_traj = []
        simulation_ended = False
        steps = 0
        
        if not show_plots:
            while not simulation_ended:
                if not env.is_goal_reached() and not antimissile.intercepted:
                    best_action_node = mcts.search()
                    env.step(best_action_node.state - env.get_state())
                    missile_traj.append(env.get_state())
                    
                    antimissile.proportional_navigation()
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
                nonlocal simulation_ended, missile_reached_goal, missile_intercepted
                if not simulation_ended:
                    if not env.is_goal_reached() and not antimissile.intercepted:
                        best_action_node = mcts.search()
                        env.step(best_action_node.state - env.get_state())
                        missile_traj.append(env.get_state())
                        
                        antimissile.proportional_navigation()
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

                if antimissile.intercepted:
                    ax.scatter(*antimissile.intercept_point, color='green', marker='X', s=200, label='Collision')

                if env.is_goal_reached():
                    ax.scatter(*env.goal, color='red', marker='X', s=200, label='Goal Reached')

                ax.legend()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            def animate():
                ani = FuncAnimation(fig, update, frames=100, interval=200, blit=False)
                plt.show()

            animate()

    return missile_reached_goal, missile_intercepted, num_simulations - missile_reached_goal - missile_intercepted

def main(show_plots=False, num_simulations=100, max_steps=10000, missile_position=None, antimissile_position=None, goal_position=None, missile_velocity=100, antimissile_velocity=105, weighting_factor=0.5):
    
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

    print(f"\nSimulation Results for MCTS planner:")
    print(f"Missile reached goal in {missile_reached_goal} out of 100 simulations")
    print(f"Missile intercepted in {missile_intercepted} out of 100 simulations")
    print(f"Number of simulations that reached max steps: {max_steps_reached}")
    print(f"Missile Success Rate: {round(missile_reached_goal / num_simulations * 100)}% and Anti-Missile Success Rate: {round((num_simulations - missile_reached_goal) / num_simulations * 100)}%\n")

if __name__ == "__main__":
    main()