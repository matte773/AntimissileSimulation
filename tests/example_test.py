# import pytest
# import sys
# from antimissilesim import mcts as MCTS
# from antimissilesim import pomdp as POMDP
# from antimissilesim import kalman as Kalman

# # @pytest.mark.skipif(sys.platform == 'win32', reason="Skipping Windows-specific tests on non-Windows platforms")

# def test_import():
#     """Ensure the package is importable."""
#     try:
#         # Attempt to import the package
#         # This will raise an ImportError if the package is not found
#         print("Importing antimissilesim...")
#         import antimissilesim
#         print("Import successful.")
#         print("Modules available:")
#         print("mcts:", MCTS)
#         print("pomdp:", POMDP)
#         print("kalman:", Kalman)

#         print("\nTest Passed ✅: Package imported successfully.")
#         print
#     except ImportError:
#         print("\nTest Failed ❌: Package failed to import.")
#         pytest.fail("Package failed to import")

# def test_mcts():
#     """Test that Monte Carlo Tree Search runs without errors."""
#     mcts = MCTS()
#     action = mcts.run()
#     assert action is not None, "MCTS returned no action"

# def test_pomdp():
#     """Test that POMDP runs without errors."""
#     pomdp = POMDP()
#     action = pomdp.run()
#     assert action is not None, "POMDP returned no action"

# def test_kalman_filter():
#     """Test that Kalman filtering updates state correctly."""
#     kf = Kalman()
#     state = kf.predict()
#     assert state is not None, "Kalman filter returned no prediction"

# def test_cli_kalman():
#     """Test CLI execution for Kalman mode."""
#     # Add a platform check to ensure it's not running pywin32-specific code on Linux
#     if sys.platform != "win32":  # Skip Windows-specific tests on Linux
#         result = subprocess.run(["python", "-m", "antimissilesim.cli", "--mode", "kalman"], capture_output=True)
#         assert result.returncode == 0, "CLI execution failed for Kalman mode"

# def test_cli_mcts():
#     """Test CLI execution for MCTS mode."""
#     if sys.platform != "win32":  # Skip Windows-specific tests on Linux
#         result = subprocess.run(["python", "-m", "antimissilesim.cli", "--mode", "mcts"], capture_output=True)
#         assert result.returncode == 0, "CLI execution failed for MCTS mode"

# def test_cli_pomdp():
#     """Test CLI execution for POMDP mode."""
#     if sys.platform != "win32":  # Skip Windows-specific tests on Linux
#         result = subprocess.run(["python", "-m", "antimissilesim.cli", "--mode", "pomdp"], capture_output=True)
#         assert result.returncode == 0, "CLI execution failed for POMDP mode"import pytest
import sys
import pytest
import subprocess
from antimissilesim import mcts as MCTS
from antimissilesim import pomdp as POMDP
from antimissilesim import kalman as Kalman

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason="Tests in this module do not run on Windows")

def test_import():
    try:
        import antimissilesim
        print("Import successful.")
        print("Modules available:")
        print("mcts:", MCTS)
        print("pomdp:", POMDP)
        print("kalman:", Kalman)
        print("\nTest Passed ✅: Package imported successfully.")
    except ImportError:
        print("\nTest Failed ❌: Package failed to import.")
        pytest.fail("Package failed to import")

def test_mcts():
    result = subprocess.run(
        ["python", "-m", "antimissilesim", "--mode", "mcts"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"MCTS failed with error: {result.stderr}"

def test_pomdp():
    result = subprocess.run(
        ["python", "-m", "antimissilesim", "--mode", "pomdp"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"POMDP failed with error: {result.stderr}"

# def test_kalman():
#     result = subprocess.run(
#         ["python", "-m", "antimissilesim", "--mode", "kalman"],
#         capture_output=True,
#         text=True
#     )
#     assert result.returncode == 0, f"Kalman failed with error: {result.stderr}"

# @pytest.mark.parametrize("flag, value", [
#     ("--show_plots", ""),
#     ("--num_simulations", "500"),
#     ("--max_steps", "2000"),
#     ("--missile_position", "0 0 100"),
#     ("--antimissile_position", "10 10 200"),
#     ("--goal_position", "50 50 0"),
#     ("--missile_velocity", "150"),
#     ("--antimissile_velocity", "120"),
#     ("--weighting_factor", "0.6"),
# ])
# def test_mcts_flags(flag, value):
#     result = subprocess.run(
#         ["python", "-m", "antimissilesim", "--mode", "mcts", flag, value],
#         capture_output=True,
#         text=True
#     )
#     assert result.returncode == 0, f"MCTS with {flag}={value} failed with error: {result.stderr}"

# @pytest.mark.parametrize("flag, value", [
#     ("--show_plots", ""),
#     ("--num_simulations", "500"),
#     ("--max_steps", "2000"),
#     ("--missile_position", "0 0 100"),
#     ("--antimissile_position", "10 10 200"),
#     ("--goal_position", "50 50 0"),
#     ("--missile_velocity", "150"),
#     ("--antimissile_velocity", "120"),
#     ("--weighting_factor", "0.6"),
# ])
# def test_pomdp_flags(flag, value):
#     result = subprocess.run(
#         ["python", "-m", "antimissilesim", "--mode", "pomdp", flag, value],
#         capture_output=True,
#         text=True
#     )
#     assert result.returncode == 0, f"POMDP with {flag}={value} failed with error: {result.stderr}"

# @pytest.mark.parametrize("flag, value", [
#     ("--show_plots", ""),
#     ("--num_simulations", "500"),
#     ("--max_steps", "2000"),
#     ("--missile_position", "0 0 100"),
#     ("--antimissile_position", "10 10 200"),
#     ("--goal_position", "50 50 0"),
#     ("--missile_velocity", "150"),
#     ("--antimissile_velocity", "120"),
#     ("--weighting_factor", "0.6"),
# ])

# def test_kalman_flags(flag, value):
#     result = subprocess.run(
#         ["python", "-m", "antimissilesim", "--mode", "kalman", flag, value],
#         capture_output=True,
#         text=True
#     )
#     assert result.returncode == 0, f"Kalman with {flag}={value} failed with error: {result.stderr}"

# if __name__ == "__main__":
#     pytest.main()

# if __name__ == "__main__":
#     # pytest.main()
#     # Run the tests
#     test_import()