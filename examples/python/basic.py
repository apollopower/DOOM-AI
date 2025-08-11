#!/usr/bin/env python3

#####################################################################
# Enhanced ViZDoom script with Deep Q-Learning capabilities
# This script can run in two modes:
# 1. Random mode (original behavior) - set ENABLE_LEARNING = False  
# 2. DQN Learning mode - set ENABLE_LEARNING = True
#
# The DQN agent learns to play using neural networks and experience replay
#####################################################################

from __future__ import print_function
import vizdoom as vzd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import skimage.color, skimage.transform
from random import choice, sample, randint, random
from time import sleep, time

# Action constants - makes actions much more readable
# Each action corresponds to: [MOVE_LEFT, MOVE_RIGHT, ATTACK]
ACTION_MOVE_LEFT = 0
ACTION_MOVE_RIGHT = 1  
ACTION_ATTACK = 2
ACTION_NAMES = ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]

# Learning configuration
ENABLE_LEARNING = True  # Set to False for original random behavior
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.99
EPOCHS = 5  # Reduced for quicker demo
LEARNING_STEPS_PER_EPOCH = 1000
REPLAY_MEMORY_SIZE = 5000
BATCH_SIZE = 32
RESOLUTION = (30, 45)  # Downsampled resolution for neural network
FRAME_REPEAT = 12
MODEL_SAVEFILE = "./basic_model.pth"


def get_action_array(action_index):
    """Convert action index to boolean array for ViZDoom"""
    actions = [
        [True, False, False],   # ACTION_MOVE_LEFT
        [False, True, False],   # ACTION_MOVE_RIGHT  
        [False, False, True],   # ACTION_ATTACK
    ]
    return actions[action_index]


def get_action_name(action_index):
    """Get human-readable name for action"""
    return ACTION_NAMES[action_index]


def preprocess(img):
    """Converts and down-samples the input image for neural network processing"""
    img = skimage.color.rgb2gray(img)
    img = skimage.transform.resize(img, RESOLUTION)
    img = img.astype(np.float32)
    return img


class ReplayMemory:
    """Experience replay buffer for storing and sampling transitions"""
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, channels, RESOLUTION[0], RESOLUTION[1])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)
        
        self.capacity = capacity
        self.size = 0
        self.pos = 0
    
    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, 0, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, 0, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


class DQN(nn.Module):
    """Deep Q-Network architecture"""
    def __init__(self, available_actions_count):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, available_actions_count)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 192)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_best_action(model, state):
    """Get the best action according to the current policy"""
    state = torch.from_numpy(state)
    q_values = model(state)
    _, action_idx = torch.max(q_values, 1)
    return action_idx.item()


def learn_from_memory(model, optimizer, memory, criterion):
    """Train the network on a batch of experiences from replay memory"""
    if memory.size > BATCH_SIZE:
        s1, a, s2, isterminal, r = memory.get_sample(BATCH_SIZE)
        
        s1_tensor = torch.from_numpy(s1)
        s2_tensor = torch.from_numpy(s2)
        
        # Get Q-values for current and next states
        current_q_values = model(s1_tensor)
        next_q_values = model(s2_tensor).detach()
        
        # Calculate target Q-values
        target_q_values = current_q_values.clone()
        for i in range(BATCH_SIZE):
            if isterminal[i]:
                target_q_values[i][a[i]] = r[i]
            else:
                target_q_values[i][a[i]] = r[i] + DISCOUNT_FACTOR * torch.max(next_q_values[i])
        
        # Train the network
        loss = criterion(current_q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def exploration_rate(epoch):
    """Epsilon-greedy exploration rate that decreases over time"""
    start_eps = 1.0
    end_eps = 0.1
    decay_epochs = EPOCHS * 0.8
    
    if epoch < decay_epochs:
        return start_eps - (epoch / decay_epochs) * (start_eps - end_eps)
    else:
        return end_eps


if __name__ == "__main__":
    print("=== Enhanced ViZDoom Basic Agent ===")
    print(f"Learning Mode: {'ENABLED' if ENABLE_LEARNING else 'DISABLED (Random Agent)'}")
    
    # Create DoomGame instance. It will run the game and communicate with you.
    game = vzd.DoomGame()

    # Now it's time for configuration!
    # load_config could be used to load configuration instead of doing it here with code.
    # If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
    game.load_config("../../scenarios/basic.cfg")

    # Sets path to additional resources wad file which is basically your scenario wad.
    # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
    game.set_doom_scenario_path("../../scenarios/basic.wad")

    # Sets map to start (scenario .wad files can contain many maps).
    game.set_doom_map("map01")

    # Sets resolution. Default is 320X240
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Sets the screen buffer format. Use GRAY8 for learning mode, RGB24 for display mode
    if ENABLE_LEARNING:
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.set_window_visible(False)  # Faster training without visual window
    else:
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_window_visible(True)

    # Enables depth buffer.
    game.set_depth_buffer_enabled(True)

    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(True)

    # Enables buffer with top down map of the current episode/level.
    game.set_automap_buffer_enabled(True)

    # Enables information about all objects present in current episode/level.
    game.set_objects_info_enabled(True)

    # Enables information about all sectors (map layout).
    game.set_sectors_info_enabled(True)

    # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)  # If hud is enabled
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)  # Bullet holes and blood on the walls
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)  # Smoke and blood
    game.set_render_messages(False)  # In-game messages
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

    # Adds buttons that will be allowed.
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.ATTACK)

    # Adds game variables that will be included in state.
    game.add_available_game_variable(vzd.GameVariable.AMMO2)

    # Causes episodes to finish after 200 tics (actions)
    game.set_episode_timeout(200)

    # Makes episodes start after 10 tics (~after raising the weapon)
    game.set_episode_start_time(10)

    # Window visibility already set above based on learning mode

    # Turns on the sound. (turned off by default)
    game.set_sound_enabled(False)

    # Sets the livin reward (for each move) to -1
    game.set_living_reward(-1)

    # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
    game.set_mode(vzd.Mode.PLAYER)


    # Enables engine output to console.
    # game.set_console_enabled(True)

    # Show the HUD, by default this is False
    # game.set_render_hud(True)

    # Initialize the game. Further configuration won't take any effect from now on.
    game.init()

    # Define available actions using constants for better readability
    # Available actions: MOVE_LEFT, MOVE_RIGHT, ATTACK
    n_actions = 3  # Total number of actions available
    
    if ENABLE_LEARNING:
        # Initialize learning components
        print("Initializing Deep Q-Network...")
        model = DQN(n_actions)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        
        print(f"Training for {EPOCHS} epochs...")
        start_time = time()
        
        # Training loop
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            epoch_scores = []
            
            game.new_episode()
            
            # Training steps for this epoch
            for step in range(LEARNING_STEPS_PER_EPOCH):
                if game.is_episode_finished():
                    epoch_scores.append(game.get_total_reward())
                    game.new_episode()
                
                # Get current state
                state = game.get_state()
                screen = preprocess(state.screen_buffer)
                
                # Choose action using epsilon-greedy policy
                eps = exploration_rate(epoch)
                if random() < eps:
                    action_idx = randint(0, n_actions - 1)  # Random exploration
                else:
                    screen_input = screen.reshape([1, 1, RESOLUTION[0], RESOLUTION[1]])
                    action_idx = get_best_action(model, screen_input)  # Exploit learned policy
                
                # Execute action and get reward
                action_array = get_action_array(action_idx)
                reward = game.make_action(action_array, FRAME_REPEAT)
                is_terminal = game.is_episode_finished()
                
                # Get next state
                if not is_terminal:
                    next_screen = preprocess(game.get_state().screen_buffer)
                else:
                    next_screen = None
                
                # Store transition in replay memory
                memory.add_transition(screen, action_idx, next_screen, is_terminal, reward)
                
                # Learn from experience
                learn_from_memory(model, optimizer, memory, criterion)
            
            # Epoch summary
            if epoch_scores:
                avg_score = np.mean(epoch_scores)
                print(f"Episodes completed: {len(epoch_scores)}")
                print(f"Average score: {avg_score:.2f}")
                print(f"Exploration rate: {eps:.3f}")
            
            # Save model periodically
            if (epoch + 1) % 2 == 0:
                torch.save(model.state_dict(), MODEL_SAVEFILE)
                print(f"Model saved to {MODEL_SAVEFILE}")
        
        training_time = time() - start_time
        print(f"\nTraining completed in {training_time/60:.1f} minutes!")
        
        # Test the trained model
        print("\nTesting trained model...")
        game.set_window_visible(True)  # Show the game window for testing
        
        test_episodes = 3
        for i in range(test_episodes):
            print(f"\nTest Episode {i + 1}")
            game.new_episode()
            
            while not game.is_episode_finished():
                state = game.get_state()
                screen = preprocess(state.screen_buffer)
                screen_input = screen.reshape([1, 1, RESOLUTION[0], RESOLUTION[1]])
                
                # Use trained policy (no exploration)
                action_idx = get_best_action(model, screen_input)
                action_array = get_action_array(action_idx)
                reward = game.make_action(action_array, FRAME_REPEAT)
                
                sleep(0.02)  # Slow down for viewing
            
            print(f"Test Episode {i + 1} Score: {game.get_total_reward()}")
    
    else:
        # Original random agent behavior
        episodes = 10

    # Sets time that will pause the engine after each action (in seconds)
    # Without this everything would go too fast for you to keep track of what's happening.
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    for i in range(episodes):
        print("Episode #" + str(i + 1))

        # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
        game.new_episode()

        while not game.is_episode_finished():

            # Gets the state
            state = game.get_state()

            # Which consists of:
            n = state.number
            vars = state.game_variables
            screen_buf = state.screen_buffer
            depth_buf = state.depth_buffer
            labels_buf = state.labels_buffer
            automap_buf = state.automap_buffer
            labels = state.labels
            objects = state.objects
            sectors = state.sectors

            # Games variables can be also accessed via:
            #game.get_game_variable(GameVariable.AMMO2)

            # Makes a random action and get reward
            random_action = randint(0, n_actions - 1)  # Choose random action index
            action_array = get_action_array(random_action)  # Convert to boolean array
            r = game.make_action(action_array)
            
            # Show which action was taken (for educational purposes)
            action_name = get_action_name(random_action)

            # Makes a "prolonged" action and skip frames:
            # skiprate = 4
            # r = game.make_action(choice(actions), skiprate)

            # The same could be achieved with:
            # game.set_action(choice(actions))
            # game.advance_action(skiprate)
            # r = game.get_last_reward()

            # Prints state's game variables, action taken, and reward.
            print("State #" + str(n))
            print("Action taken:", action_name)
            print("Game variables:", vars)
            print("Reward:", r)
            print("=====================")

            if sleep_time > 0:
                sleep(sleep_time)

        # Check how the episode went.
        print("Episode finished.")
        print("Total reward:", game.get_total_reward())
        print("************************")

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()
