import sys
import pytest
import logging
import subprocess
from antimissilesim import mcts as MCTS
from antimissilesim import pomdp as POMDP
from antimissilesim import kalman as Kalman

# pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason="Tests in this module do not run on Windows")

# Test that each mode imports successfully
def test_import():
    """Test if the package imports successfully."""
    try:
        import antimissilesim
        assert hasattr(antimissilesim, 'mcts')
        assert hasattr(antimissilesim, 'pomdp')
        assert hasattr(antimissilesim, 'kalman')
    except ImportError:
        pytest.fail("Package failed to import")

# Test that Kalman runs with various flag-value pairs
@pytest.mark.parametrize("flags", [
    [("--num_simulations", "1"), ("--max_steps", "1000")],  # Default flags
    [("--num_simulations", "1"), ("--weighting_factor", "1.1")],  # Weighting factor > 1
    [("--num_simulations", "1"), ("--weighting_factor", "-0.1")],  # Negative weighting factor
    [("--num_simulations", "1"), 
     ("--missile_position", "0,0,0"),  # Use a single string with multiple values
     ("--antimissile_position", "500,500,500"),  # Same for antimissile position
     ("--goal_position", "50,750,750")],  # Same for goal position
    [("--num_simulations", "-1")], # Negative number of simulations
    [("--num_simulations", "1"), ("--missile_velocity", "110"), ("--antimissile_velocity", "120")],  # Different velocities

])
def test_kalman_flags(flags):
    """Test if Kalman runs successfully with multiple flag-value pairs."""
    # Flatten the list of flag-value pairs into a single list of arguments
    args = ["python", "-m", "antimissilesim", "--mode", "kalman"]
    
    for flag, value in flags:
        if flag in ["--missile_position", "--antimissile_position", "--goal_position"]:
            # Handle multi-value flags by splitting the string into individual components
            value = value.split(",")
            args.extend([flag] + value)
        else:
            args.extend([flag, value])

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=30  # Increased timeout to 60 seconds
    )

    # Check the captured output for any errors or prompts
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success: {result.stdout}")

    assert result.returncode == 0, f"Kalman with flags {flags} failed with error: {result.stderr}"

# Test that MCTS runs with various flag-value pairs
@pytest.mark.parametrize("flags", [
    [("--num_simulations", "1"), ("--max_steps", "1000")],  # Default flags
    [("--num_simulations", "1"), ("--weighting_factor", "1.1")],  # Weighting factor > 1
    [("--num_simulations", "1"), ("--weighting_factor", "-0.1")],  # Negative weighting factor
    [("--num_simulations", "1"), 
     ("--missile_position", "0,0,0"),  # Use a single string with multiple values
     ("--antimissile_position", "500,500,500"),  # Same for antimissile position
     ("--goal_position", "50,750,750")],  # Same for goal position
    [("--num_simulations", "-1")], # Negative number of simulations
    [("--num_simulations", "1"), ("--missile_velocity", "110"), ("--antimissile_velocity", "120")],  # Different velocities

])
def test_mcts_flags(flags):
    """Test if MCTS runs successfully with multiple flag-value pairs."""
    # Flatten the list of flag-value pairs into a single list of arguments
    args = ["python", "-m", "antimissilesim", "--mode", "mcts"]
    
    for flag, value in flags:
        if flag in ["--missile_position", "--antimissile_position", "--goal_position"]:
            # Handle multi-value flags by splitting the string into individual components
            value = value.split(",")
            args.extend([flag] + value)
        else:
            args.extend([flag, value])

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=30  # Increased timeout to 60 seconds
    )

    # Check the captured output for any errors or prompts
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success: {result.stdout}")

    assert result.returncode == 0, f"MCTS with flags {flags} failed with error: {result.stderr}"

# Test that POMDP runs with various flag-value pairs
@pytest.mark.parametrize("flags", [
    [("--num_simulations", "1"), ("--max_steps", "1000")],  # Default flags
    [("--num_simulations", "1"), ("--weighting_factor", "1.1")],  # Weighting factor > 1
    [("--num_simulations", "1"), ("--weighting_factor", "-0.1")],  # Negative weighting factor
    [("--num_simulations", "1"), 
     ("--missile_position", "0,0,0"),  # Use a single string with multiple values
     ("--antimissile_position", "500,500,500"),  # Same for antimissile position
     ("--goal_position", "50,750,750")],  # Same for goal position
    [("--num_simulations", "-1")], # Negative number of simulations
    [("--num_simulations", "1"), ("--missile_velocity", "110"), ("--antimissile_velocity", "120")],  # Different velocities

])
def test_pomdp_flags(flags):
    """Test if POMDP runs successfully with multiple flag-value pairs."""
    # Flatten the list of flag-value pairs into a single list of arguments
    args = ["python", "-m", "antimissilesim", "--mode", "pomdp"]
    
    for flag, value in flags:
        if flag in ["--missile_position", "--antimissile_position", "--goal_position"]:
            # Handle multi-value flags by splitting the string into individual components
            value = value.split(",")
            args.extend([flag] + value)
        else:
            args.extend([flag, value])

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=30  # Increased timeout to 60 seconds
    )

    # Check the captured output for any errors or prompts
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success: {result.stdout}")

    assert result.returncode == 0, f"POMDP with flags {flags} failed with error: {result.stderr}"
    
if __name__ == "__main__":
    pytest.main()
