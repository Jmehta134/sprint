"""
===================================================
    eLSI Sprint 1 - Task 1B : Q-Learning
===================================================

Participant template.

HOW TO RUN
  1. Open the Task 1B scene in CoppeliaSim.
  2. Start the bridge:   python3 bridge_task1b.py --eval
  3. Train:              python3 task1b_template.py --mode train
     Test (no learning): python3 task1b_template.py --mode test

MODES
  train : choose actions with exploration AND update the Q-table.
          The Q-table is saved to disk on exit.
  test  : load the saved Q-table, act greedily, and DO NOT update it.

WHAT YOU IMPLEMENT
  get_state()     - how to turn the 5 sensor values into a discrete state.
  get_reward()    - how good the latest reading is.
  choose_action() - which action to take in a given state (the policy).

Team ID: [ 782 ]
"""

import time
import os
import pickle
import random
import argparse

from connector_task1b import CoppeliaClient

# The five line sensors, ordered left -> right across the robot ([0.0, 1.0]).
SENSOR_ORDER = ['left_corner', 'left', 'middle', 'right', 'right_corner']

# Action set: index -> (left_speed, right_speed). 
ACTIONS = [
    (4.0,4.0),     # straight
    (3.4,4.6),     # slight left
    (4.6,3.4),     # slight right
    (2.6,5.4),     # hard left
    (5.4,2.6),     # hard right
    (1.0,5.0),    # hardest left
    (5.0,1.0),    # hardest right
]



# Hyper parameter for tuning
ALPHA = 0.2
GAMMA = 0.95
EPSILON = 0.15

# Saved next to this script, so it doesn't depend on the launch directory.
Q_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.pkl")


# =============================================================================
#  TODO (participants): implement get_state(), get_reward() and choose_action().
#  You may also add your own helper functions in this section.
# =============================================================================
import random

BACKGROUND_IS_HIGH = False
LAST_KNOWN_LINE = (0, 0, 1, 0, 0) 

def get_state(sensors):
    global BACKGROUND_IS_HIGH, LAST_KNOWN_LINE
    
    left_corner = sensors['left_corner']
    right_corner = sensors['right_corner']
    
    if left_corner < 0.4 and right_corner < 0.4:
        BACKGROUND_IS_HIGH = False
    elif left_corner > 0.6 and right_corner > 0.6:
        BACKGROUND_IS_HIGH = True
        
    current_reading = []
    for sensor in SENSOR_ORDER:
        val = sensors[sensor]
        if BACKGROUND_IS_HIGH:
            current_reading.append(1 if val < 0.5 else 0)
        else:
            current_reading.append(1 if val > 0.5 else 0)
                
    current_tuple = tuple(current_reading)
    
    # --- THE MAGIC HAPPENS HERE ---
    if current_tuple == (0, 0, 0, 0, 0):
        # We are lost! Append the memory to the state so the Q-table knows how to recover.
        return ("LOST", LAST_KNOWN_LINE)
    else:
        # We are on the line. Update memory and just return the current 1D state.
        LAST_KNOWN_LINE = current_tuple
        return current_tuple

def get_reward(sensors, state):
    # Check if the state is our special 2D "LOST" state
    if state[0] == "LOST":
        return -30
        
    # Otherwise, it is a standard 1D state tuple
    if state == (0, 0, 1, 0, 0):
        return 10
    elif state in [(0, 1, 1, 0, 0), (0, 0, 1, 1, 0)]:
        return 7
    elif state in [(0, 1, 0, 0, 0), (0, 0, 0, 1, 0)]:
        return 4
    elif state in [(1, 1, 0, 0, 0), (0, 0, 0, 1, 1)]:
        return 2
    else:
        return -5

def choose_action(agent, state, training):
    agent._ensure(state)
    
    if training:
        if random.random() < agent.epsilon:
            # Smart Exploration when lost
            if state[0] == "LOST":
                last_known = state[1]
                # If we fell off to the right (saw line on left sensors previously)
                if last_known[0] == 1 or last_known[1] == 1:
                    return 3 # Force a hard left turn
                # If we fell off to the left (saw line on right sensors previously)
                elif last_known[3] == 1 or last_known[4] == 1:
                    return 4 # Force a hard right turn
                else:
                    return random.choice([3, 4])
            
            # Normal random exploration if we are on the line
            return random.randint(0, agent.n_actions - 1)

    q_values = agent.q_table[state]
    return q_values.index(max(q_values))

# =============================================================================
#  Q-learning agent (Don't Edit this)
# =============================================================================
class QLearningAgent:
    def __init__(self, n_actions, alpha, gamma, epsilon, path):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.path = path
        self.q_table = {}   

    def _ensure(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.n_actions

    def update(self, state, action, reward, next_state):
        """Q-learning update. Called only in train mode."""
        self._ensure(state)
        self._ensure(next_state)
        best_next = max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"Loaded Q-table ({len(self.q_table)} states) from {self.path}")
            return True
        return False

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Saved Q-table ({len(self.q_table)} states) to {self.path}")


# =============================================================================
#  Main loop
# =============================================================================
def run(mode):
    training = (mode == "train")

    agent = QLearningAgent(len(ACTIONS), ALPHA, GAMMA, EPSILON, Q_TABLE_PATH)
    loaded = agent.load()
    if not training and not loaded:
        print("ERROR: test mode needs a trained Q-table. Run --mode train first.")
        return

    client = CoppeliaClient(host="127.0.0.1", port=50002)
    client.connect()
    print(f"Connected to bridge_task1b. Mode = {mode}. (Ctrl+C to stop)")

    try:
        print("\nPython script running continuously. Waiting for valid sensor data...")
        prev_state = None
        prev_action = None
        step_count = 0

        while True:
            sensors = client.receive_sensor_data()
            
            # Check for stop or pause (None or all zeros)
            if sensors is None or all(v == 0.0 for v in sensors.values()):
                # Reset previous state so we don't carry over Q-updates between episodes!
                prev_state = None
                prev_action = None
                time.sleep(0.05)
                continue

            state = get_state(sensors)
            reward = get_reward(sensors, state)
            
            if training and prev_state is not None:
                agent.update(prev_state, prev_action, reward, state)

            action = choose_action(agent, state, training)
            left, right = ACTIONS[action]
            
            client.send_motor_command(
                left, right,
                state=list(state),  
                reward=reward,
                action=action,
            )

            prev_state, prev_action = state, action
            
            # Save periodically (e.g., every 1000 steps = ~50 seconds) to avoid lagging the simulation
            if training:
                step_count += 1
                if step_count % 1000 == 0:
                    print(f"\n[Auto-Save] Saving Q-Table at step {step_count}...")
                    agent.save()

            time.sleep(0.05)   # ~20 Hz
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            client.send_motor_command(0.0, 0.0, state=0, reward=0.0, action=0)
        except Exception:
            pass
        client.close()
        if training:
            agent.save()   # persist what was learned

def main():
    parser = argparse.ArgumentParser(description="Task 1B - Q-Learning")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="train: explore + update Q-table; test: greedy, no update")
    args = parser.parse_args()
    
    run(args.mode)

if __name__ == "__main__":
    main()