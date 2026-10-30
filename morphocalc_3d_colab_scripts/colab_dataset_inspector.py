"""
================================================================================
MorphoCalc 3D: Download & Inspect Real CFD Data
================================================================================
INSTRUCTIONS:
1. Open a new Google Colab notebook.
2. In the FIRST cell, copy and paste this exact command and hit run:
   !pip install datasets

3. In the SECOND cell, copy and paste all the code below and hit run.
"""

from datasets import load_dataset
import numpy as np

print("Connecting to Hugging Face Global Servers...")
print("Streaming AhmedML 3D CFD Dataset...\n")

# We use streaming=True so we don't have to wait for a 10GB download right now.
# It acts like a straw, sipping just the first simulation so we can look at it.
try:
    dataset = load_dataset("neashton/ahmedml", streaming=True)
    
    # Grab the very first 3D aerodynamic simulation in the dataset
    first_simulation = next(iter(dataset['train']))
    
    print("=== DATASET METADATA ===")
    print("Keys inside the dataset:", list(first_simulation.keys()))
    
    print("\n=== RAW DATA EXTRACTED ===")
    # Loop through the data to see the shapes of the tensors
    for key in first_simulation.keys():
        data = first_simulation[key]
        if isinstance(data, list):
            data_np = np.array(data)
            print(f"Feature: '{key}' | Shape: {data_np.shape} | This is a Tensor!")
        else:
            # If it's a giant block of bytes/numbers, just show the size
            print(f"Feature: '{key}' | Size: {len(str(data))} bytes of data")

    print("\nSuccess! You have successfully tapped into a real 3D CFD dataset.")
    print("Notice how the physical geometry is just stored as massive grids of numbers? That is what we have to clean into X,Y,Z tensors.")

except Exception as e:
    print("An error occurred while streaming the dataset. HuggingFace servers might be busy.")
    print(f"Error details: {e}")
