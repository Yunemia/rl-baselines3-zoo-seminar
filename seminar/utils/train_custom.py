import os
import sys

# Add the project root (/seminar) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl_zoo3.train import train

if __name__ == "__main__":
    train()
