"""
SIMPLE TEST VERSION - Check if basic execution works
"""

print("=" * 80)
print("TESTING: Script is executing...")
print("=" * 80)

import sys
print(f"\nPython version: {sys.version}")

try:
    import pandas as pd
    print("✓ pandas imported")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✓ numpy imported")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported")
except ImportError as e:
    print(f"✗ matplotlib import failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("All imports successful!")
print("=" * 80)

# Simple menu test
print("\nMENU TEST:")
print("1. Option 1")
print("2. Option 2")
print("3. Exit")

choice = input("\nEnter choice (1-3): ").strip()
print(f"\nYou selected: {choice}")

if choice == "1":
    print("\nRunning option 1...")
    print("Creating sample data...")
    
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'value': range(100)
    })
    
    print(f"✓ Created DataFrame with {len(data)} rows")
    print(data.head())
    
elif choice == "2":
    print("\nRunning option 2...")
    print("Testing numpy...")
    arr = np.array([1, 2, 3, 4, 5])
    print(f"✓ Created array: {arr}")
    print(f"  Mean: {arr.mean()}")
    
else:
    print("\nExiting...")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)