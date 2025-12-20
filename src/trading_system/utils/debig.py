"""
Debug script to check RL library availability
Run this to diagnose the import issue
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("\n" + "="*60)

# Test 1: Check gymnasium
print("\n[1/4] Testing gymnasium import...")
try:
    import gymnasium as gym
    from gymnasium import spaces
    print(f"✓ gymnasium imported successfully")
    print(f"  Version: {gym.__version__}")
    print(f"  Location: {gym.__file__}")
    GYMNASIUM_OK = True
except ImportError as e:
    print(f"✗ gymnasium import failed: {e}")
    GYMNASIUM_OK = False
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    GYMNASIUM_OK = False

# Test 2: Check stable-baselines3
print("\n[2/4] Testing stable-baselines3 import...")
try:
    from stable_baselines3 import PPO, A2C
    import stable_baselines3
    print(f"✓ stable-baselines3 imported successfully")
    print(f"  Version: {stable_baselines3.__version__}")
    print(f"  Location: {stable_baselines3.__file__}")
    SB3_OK = True
except ImportError as e:
    print(f"✗ stable-baselines3 import failed: {e}")
    print(f"  This might be due to missing torch dependency")
    SB3_OK = False
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    SB3_OK = False

# Test 3: Check dependencies
print("\n[3/4] Checking dependencies...")
try:
    import numpy
    print(f"✓ numpy {numpy.__version__}")
except:
    print("✗ numpy missing")

try:
    import pandas
    print(f"✓ pandas {pandas.__version__}")
except:
    print("✗ pandas missing")

try:
    import torch
    print(f"✓ torch {torch.__version__}")
except:
    print("✗ torch missing (required by stable-baselines3)")

# Test 4: Simulate your code's import logic
print("\n[4/4] Testing your code's import pattern...")
RL_AVAILABLE = False
SB3_AVAILABLE = False

try:
    import gymnasium as gym
    RL_AVAILABLE = True
    print("✓ RL_AVAILABLE = True")
except ImportError:
    print("✗ RL_AVAILABLE = False")

try:
    from stable_baselines3 import PPO, A2C
    SB3_AVAILABLE = True
    print("✓ SB3_AVAILABLE = True")
except ImportError:
    print("✗ SB3_AVAILABLE = False")

# Final verdict
print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

if GYMNASIUM_OK and SB3_OK:
    print("✓ Both libraries are working!")
    print("  Your RL training should work.")
    print("\nIf you're still seeing the error, the issue might be:")
    print("  1. You're running a different Python environment")
    print("  2. The imports are being cached")
    print("  3. There's a circular import issue")
elif not GYMNASIUM_OK:
    print("✗ gymnasium is NOT working")
    print("  Fix: pip install gymnasium")
elif not SB3_OK:
    print("✗ stable-baselines3 is NOT working")
    print("  Common causes:")
    print("    - Missing torch: pip install torch")
    print("    - Version mismatch: pip install stable-baselines3 --upgrade")
    print("  Fix: pip install stable-baselines3")
else:
    print("✗ Unknown issue")

print("\n" + "="*60)
print("RECOMMENDED FIXES:")
print("="*60)

if not GYMNASIUM_OK or not SB3_OK:
    print("\n1. Reinstall both packages:")
    print("   pip uninstall gymnasium stable-baselines3 -y")
    print("   pip install gymnasium stable-baselines3")
    print("\n2. If that fails, install with torch:")
    print("   pip install torch gymnasium stable-baselines3")
    print("\n3. Check you're using the right Python:")
    print(f"   which python  # Should match: {sys.executable}")