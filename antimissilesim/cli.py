# # antimissilesim/cli.py
# import argparse
# from .kalman import main as kalman_main  

# def main():
#     parser = argparse.ArgumentParser(description="Run Antimissile Simulation")
#     parser.add_argument("--mode", type=str, choices=["kalman", "mcts", "pomdp"], default="kalman", help="Select mode")
#     args = parser.parse_args()

#     if args.mode == "kalman":
#         kalman_main()  # Call your Kalman filter function
#     elif args.mode == "mcts":
#         from .mcts import main as mcts_main
#         mcts_main()
#     elif args.mode == "pomdp":
#         from .pomdp import main as pomdp_main
#         pomdp_main()

# if __name__ == "__main__":
#     main()

import argparse
from .kalman import main as kalman_main  
from .mcts import main as mcts_main
from .pomdp import main as pomdp_main

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run Antimissile Simulation")
    parser.add_argument("--mode", type=str, choices=["kalman", "mcts", "pomdp"], default="kalman", help="Select mode")
    
    # Add other arguments that you want to pass to the simulation
    # parser.add_argument("--show_plots", type=bool, default=True, help="Enable or disable plot animation filter.")
    
    # Make sure to set the default value of show_plots to False
    parser.add_argument("--show_plots", action="store_true", help="Enable plot animation (default is False)")    
    parser.add_argument("--num_simulations", type=int, default=100, help="Number of simulations to run")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of steps per simulation")
    parser.add_argument("--missile_position", type=float, nargs=3, default=None, help="Initial missile position (x, y, z)")
    parser.add_argument("--antimissile_position", type=float, nargs=3, default=None, help="Initial antimissile position (x, y, z)")
    parser.add_argument("--goal_position", type=float, nargs=3, default=None, help="Goal position (x, y, z)")
    parser.add_argument("--missile_velocity", type=float, default=100, help="Missile velocity")
    parser.add_argument("--antimissile_velocity", type=float, default=105, help="Antimissile velocity")
    parser.add_argument("--weighting_factor", type=float, default=None, help="Evade weighting factor")

    # Parse the arguments
    args = parser.parse_args()

    # Pass the arguments to the respective mode
    if args.mode == "kalman":
        if args.weighting_factor is None:
            args.weighting_factor = 0.5
        kalman_main(args.show_plots, args.num_simulations, args.max_steps, args.missile_position, args.antimissile_position, 
                    args.goal_position, args.missile_velocity, args.antimissile_velocity, args.weighting_factor)
    elif args.mode == "mcts":
        if args.weighting_factor is None:
            args.weighting_factor = 0.3
        mcts_main(args.show_plots, args.num_simulations, args.max_steps, args.missile_position, args.antimissile_position, 
                  args.goal_position, args.missile_velocity, args.antimissile_velocity, args.weighting_factor)
    elif args.mode == "pomdp":
        if args.weighting_factor is None:
            args.weighting_factor = 0.8
        pomdp_main(args.show_plots, args.num_simulations, args.max_steps, args.missile_position, args.antimissile_position, 
                   args.goal_position, args.missile_velocity, args.antimissile_velocity, args.weighting_factor)

if __name__ == "__main__":
    main()
