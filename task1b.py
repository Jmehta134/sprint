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

Team ID: [ XXX ]
"""

import time
import os
import pickle
import random
import argparse
import math


from connector_task1b import CoppeliaClient

# The five line sensors, ordered left -> right across the robot ([0.0, 1.0]).
SENSOR_ORDER = ['left_corner', 'left', 'middle', 'right', 'right_corner']
# Q_table file.
Q_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.pkl")
# Action set: index -> (left_speed, right_speed). 
# We define 5 basic actions: Forward, Slight Left, Slight Right, Sharp Left, Sharp Right.
ACTIONS = [
    (2.0, 2.0),   # Action 0: Forward
    (0.5, 2.0),   # Action 1: Slight Left
    (2.0, 0.5),   # Action 2: Slight Right
    (-1.0, 2.0),  # Action 3: Sharp Left
    (2.0, -1.0),  # Action 4: Sharp Right
]

TRACK_BACKGROUND = None
LAST_ACTION = 0

# Hyper parameters for tuning
ALPHA = 0.2     
GAMMA = 0.9     
EPSILON = 0.1 

# =============================================================================
#  Core Logic Functions
# =============================================================================

def get_state(sensors):
    global TRACK_BACKGROUND
    threshold = 0.5
    
    raw_binary = []
    for key in ['left_corner', 'left', 'middle', 'right', 'right_corner']:
        if sensors[key] > threshold:
            raw_binary.append(1)
        else:
            raw_binary.append(0)
            
    lc, l, m, r, rc = raw_binary
    
    if lc == rc:
        TRACK_BACKGROUND = lc
    elif TRACK_BACKGROUND is None:
        TRACK_BACKGROUND = 0  
        
    bg_color = TRACK_BACKGROUND
        
    normalized_state = []
    for val in raw_binary:
        if bg_color == 1:
            normalized_state.append(1 - val)
        else:
            normalized_state.append(val)
            
    return tuple(normalized_state)


def get_reward(sensors, state):
    global LAST_ACTION
    
    lc, l, m, r, rc = state
    total_line_seen = sum(state)
    
    if total_line_seen == 0:
        return -5.0  
        
    reward = 0.0
    
    if state == (0, 0, 1, 0, 0):
        reward += 2.0   
    elif m == 1:
        reward += 1.0   
    elif l == 1 or r == 1:
        reward += 0.0   
    elif lc == 1 or rc == 1:
        reward -= 1.0   
        
    if LAST_ACTION == 0: 
        reward += 0.5   
    elif LAST_ACTION in [1, 2]: 
        reward += 0.0   
    else: 
        reward -= 0.5   
        
    return reward


def choose_action(agent, state, training):
    global LAST_ACTION
    agent._ensure(state)
    
    if training and random.random() < agent.epsilon:
        action = random.randint(0, agent.n_actions - 1)
    else:
        raw_q_values = agent.q_table[state]
        
        # Build a fresh, clean list to prevent ANY weird value reference issues
        clean_q_values = []
        for q in raw_q_values:
            try:
                # q != q is the universal fallback check for NaN
                if math.isnan(q) or math.isinf(q) or q != q:
                    clean_q_values.append(-999.9)
                else:
                    clean_q_values.append(float(q))
            except:
                clean_q_values.append(-999.9)
                
        agent.q_table[state] = clean_q_values 
            
        max_q = max(clean_q_values)
        best_actions = [i for i, q in enumerate(clean_q_values) if q == max_q]
        
        # ULTIMATE FAILSAFE: If the list is still empty, default to Action 0 (Forward)
        if not best_actions:
            best_actions = [0]
            
        action = random.choice(best_actions)
        
    LAST_ACTION = action
    return action

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

    last_sensors = None
    prev_state = None
    prev_action = None
    reward = 0.0

    # ... inside run(mode) in task1b.py ...
    NUM_EPISODES = 100          # Run 100 simulations
    EPISODE_DURATION = 4.0  # 7 seconds per simulation
    
    try:
        episodes_to_run = NUM_EPISODES if training else 1
        
        for episode in range(episodes_to_run):
            if training:
                print(f"\n--- Starting Episode {episode + 1}/{NUM_EPISODES} ---")
                
                # --- THE FIX: Network-Safe Reset Sequence ---
                # 1. Halt the robot to prevent physics glitches
                client.send_motor_command(0.0, 0.0, state=[0,0,0,0,0], reward=0.0, action=0)
                
                # 2. Force the scene to stop
                client.stop_simulation()
                
                # 3. Actively flush the TCP buffer while waiting. 
                # This ensures the stop command is actually received and processed by the bridge!
                print("Waiting for CoppeliaSim to reset the track...")
                t_end = time.time() + 3.0
                while time.time() < t_end:
                    client.receive_sensor_data()
                    time.sleep(0.01)
                
                # 4. Start the scene fresh
                client.start_simulation()
                
                # 5. Flush again while the physics engine drops the robot in
                t_end = time.time() + 1.0
                while time.time() < t_end:
                    client.receive_sensor_data()
                    time.sleep(0.01)
                
            else:
                print("\n--- Starting Continuous Test Run ---")
                client.start_simulation()

            start_time = time.time()
            last_sensors = None
            prev_state = None
            prev_action = None
            reward = 0.0

            # Run for the duration of the episode (or forever if testing)
            while (not training) or (time.time() - start_time < EPISODE_DURATION):
                sensors = client.receive_sensor_data()
                if sensors is not None:
                    last_sensors = sensors
                if last_sensors is None:
                    time.sleep(0.02)
                    continue

                state = get_state(last_sensors)
                if state is None:
                    state = (0, 0, 0, 0, 0) 
                    
                reward = get_reward(last_sensors, state)
                
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
                time.sleep(0.05)   
                
            # Episode complete! Save progress before the next one starts
            if training:
                print(f"Episode {episode + 1} complete. Saving Q-Table...")
                agent.save()
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            client.send_motor_command(0.0, 0.0, state=[0,0,0,0,0], reward=0.0, action=0)
            client.stop_simulation()
        except Exception:
            pass
        client.close()
        if training:
            agent.save()   


def main():
    parser = argparse.ArgumentParser(description="Task 1B - Q-Learning")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="train: explore + update Q-table; test: greedy, no update")
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()

