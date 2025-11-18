"""Convenience script to run experiments from project root."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.experiments.run_experiment import main

if __name__ == "__main__":
    main()

