# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

ViZDoom is a reinforcement learning research platform that allows AI agents to play DOOM using only visual information (screen pixels). The repository contains:
- Modified ZDoom engine with AI hooks
- Language bindings for Python, C++, Java, Lua, and Julia
- Deep Q-learning implementations using PyTorch
- Pre-built training scenarios

## Development Commands

### Building from Source

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip libboost-all-dev

# Python dependencies
pip install numpy

# Build ViZDoom
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON
make -j$(nproc)
```

### Installing via pip (Recommended)
```bash
# From PyPI
pip install vizdoom

# From source
pip install .
```

### Running Training Examples

```bash
# Navigate to Python examples
cd examples/python

# Train a new model with Q-learning
python pytorch_exam.py

# Test an existing trained model
python test_pytorch.py

# Run basic example
python basic.py
```

### Common Training Parameters
- Edit `config_file_path` in scripts to change scenarios
- Available scenarios: `basic.cfg`, `simpler_basic.cfg`, `rocket_basic.cfg`, `deadly_corridor.cfg`
- Modify `skip_learning=False` to enable training
- Set `load_model=True` to continue from saved model

## Architecture Overview

### Core Components

1. **ViZDoom Engine** (`src/vizdoom/`)
   - Modified ZDoom engine that exposes game state via shared memory
   - Key files:
     - `src/vizdoom/src/viz_main.cpp`: Main ViZDoom integration
     - `src/vizdoom/src/viz_game.cpp`: Game control interface
     - `src/vizdoom/src/viz_screen.cpp`: Screen buffer management
     - `src/vizdoom/src/viz_shared_memory.cpp`: IPC with Python/other bindings

2. **Python Binding** (`src/lib_python/`)
   - Uses pybind11 for C++ to Python interface
   - `ViZDoomGamePython.cpp`: Main Python API implementation
   - Exposes `DoomGame` class with methods like `make_action()`, `get_state()`, `new_episode()`

3. **Communication Architecture**
   - Game runs as separate process
   - Shared memory for frame buffers and game state
   - Message queue for commands and synchronization
   - Controller (`src/lib/ViZDoomController.cpp`) manages IPC

### Training Pipeline

1. **State Processing**
   - Game provides grayscale/color screen buffer (640x480 default)
   - `preprocess()` function downsamples to 30x45 for neural network
   - State includes screen pixels + game variables (ammo, health, etc.)

2. **Neural Network Architecture** (in examples)
   - Conv Layer 1: 8 filters, 6x6 kernel, stride 3
   - Conv Layer 2: 8 filters, 3x3 kernel, stride 2
   - FC Layer 1: 192 → 128 neurons
   - FC Layer 2: 128 → action_count neurons (Q-values)

3. **Q-Learning Implementation**
   - Experience replay buffer stores (s, a, r, s', done) transitions
   - Epsilon-greedy exploration with decay
   - Target Q-value: Q(s,a) = r + γ * max(Q(s',a'))
   - Batch training from replay memory

### Scenario System

Scenarios are defined in `.cfg` files with corresponding `.wad` level files:
- Configuration specifies available buttons, rewards, episode limits
- WAD files contain actual Doom map data
- Game variables exposed: ammunition, health, position, etc.

### Key Classes and Methods

**Python API (vizdoom module)**
- `DoomGame`: Main class for game control
  - `load_config(cfg_path)`: Load scenario configuration
  - `init()`: Initialize game engine
  - `new_episode()`: Start new game episode
  - `make_action(action, tics)`: Execute action for N game tics
  - `get_state()`: Returns GameState with screen_buffer, game_variables
  - `is_episode_finished()`: Check if episode ended
  - `get_total_reward()`: Get accumulated reward

**Training Loop Pattern**
```python
game = DoomGame()
game.load_config(scenario_cfg)
game.init()

for epoch in range(epochs):
    game.new_episode()
    while not game.is_episode_finished():
        state = preprocess(game.get_state().screen_buffer)
        action = get_action(state)  # ε-greedy or from network
        reward = game.make_action(action, frame_repeat)
        # Store transition, update network
```

## Important Notes

- The repository includes both the ViZDoom platform code and AI training examples
- Pre-trained model `model-doom.pth` exists for basic scenario
- Frame repeat (typically 12) used to reduce decision frequency
- Game runs at configurable speed (up to 7000 fps in sync mode)
- Multiple instances can run in parallel for distributed training