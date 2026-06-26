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
    (1.0,5.0),     # hardest left
    (5.0,1.0),     # hardest right
]

# --- NEW SPEED ACTIONS & LPF SETTINGS ---
SPEED_MULTIPLIERS = [0.8, 1.2, 1.6] # Slow, Medium, Fast
EMA_ALPHA = 0.6                     # Low Pass Filter strength (0.0 to 1.0). Lower is smoother.

# Hyper parameter for tuning
ALPHA = 0.2
GAMMA = 0.95
EPSILON = 0.2

# Saved next to this script, so it doesn't depend on the launch directory.
Q_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.pkl")


# =============================================================================
#  Custom Save/Load Wrappers (Preserves the QLearningAgent class untouched)
# =============================================================================
def load_dual_agents(steering_agent, speed_agent, path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            loaded_data = pickle.load(f)
            
        if isinstance(loaded_data, dict) and "steering" in loaded_data:
            steering_agent.q_table = loaded_data["steering"]
            speed_agent.q_table = loaded_data.get("speed", {})
            print(f"Loaded Dual Q-tables from {path}")
        else:
            print("Old format detected! Migrating steering data. Speed agent starting fresh.")
            steering_agent.q_table = loaded_data
            speed_agent.q_table = {}
        return True
    return False

def save_dual_agents(steering_agent, speed_agent, path):
    data_to_save = {
        "steering": steering_agent.q_table,
        "speed": speed_agent.q_table
    }
    with open(path, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"\nSaved Dual Q-tables (Steering: {len(steering_agent.q_table)}, Speed: {len(speed_agent.q_table)}) to {path}")


# =============================================================================
#  Sensor Logic & Steering Agent Implementations
# =============================================================================
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
    
    if current_tuple == (0, 0, 0, 0, 0):
        return ("LOST", LAST_KNOWN_LINE)
    else:
        LAST_KNOWN_LINE = current_tuple
        return current_tuple

def choose_action(agent, state, training):
    agent._ensure(state)
    if training:
        if random.random() < agent.epsilon:
            if state[0] == "LOST":
                last_known = state[1]
                if last_known[0] == 1 or last_known[1] == 1: return 3
                elif last_known[3] == 1 or last_known[4] == 1: return 4
                else: return random.choice([3, 4])
            return random.randint(0, agent.n_actions - 1)

    q_values = agent.q_table[state]
    return q_values.index(max(q_values))


# =============================================================================
#  Speed Agent Implementations
# =============================================================================
def get_speed_state(steering_state):
    """3-State system to prevent bang-bang oscillation."""
    if steering_state[0] == "LOST":
        return 2  # Danger / Lost
    # Safely unpacking standard 1D tuple
    elif steering_state in [(0, 0, 1, 0, 0), (0, 1, 1, 0, 0), (0, 0, 1, 1, 0), (0, 1, 1, 1, 0)]:
        return 0  # Straight & Safe
    else:
        return 1  # Curve / Drifting

def get_speed_reward(speed_state, speed_action):
    """
    speed_state: 0 (Straight), 1 (Curve), 2 (Lost/Danger)
    speed_action: 0 (Slow), 1 (Medium), 2 (Fast)
    """
    
    # 1. Survival & Recovery Rules (State 2: LOST)
    if speed_state == 2: 
        if speed_action == 2: return -40  # Death wish. Slamming gas while blind.
        if speed_action == 1: return -15  # Still too fast for safe recovery.
        if speed_action == 0: return 5    # PERFECT. Slamming the brakes to let the steering agent recover!
        
    # 2. Straightaway Rules (Force it to blast the throttle)
    if speed_state == 0:
        if speed_action == 2: return 20   # Maximum reward for max speed
        if speed_action == 1: return 2    # Mediocre reward
        if speed_action == 0: return -15  # HEAVY PENALTY for driving slow on a straight
        
    # 3. Cornering Rules (Force it to maintain momentum safely)
    if speed_state == 1:
        if speed_action == 2: return -15  # Penalty for taking a curve too fast
        if speed_action == 1: return 15   # Massive reward for taking curves at medium speed
        if speed_action == 0: return -5   # Penalty for braking too hard into a gentle curve
        
    return 0


def choose_speed_action(agent, state, training):
    agent._ensure(state)
    if training and random.random() < agent.epsilon:
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

    # Instantiate both brains
    steering_agent = QLearningAgent(len(ACTIONS), ALPHA, GAMMA, EPSILON, Q_TABLE_PATH)
    speed_agent = QLearningAgent(len(SPEED_MULTIPLIERS), ALPHA, GAMMA, EPSILON, Q_TABLE_PATH)
    
    # Custom loader
    loaded = load_dual_agents(steering_agent, speed_agent, Q_TABLE_PATH)
    
    if not training and not loaded:
        print("ERROR: test mode needs a trained Q-table. Run --mode train first.")
        return

    client = CoppeliaClient(host="127.0.0.1", port=50002)
    client.connect()
    print(f"Connected to bridge_task1b. Mode = {mode}. (Ctrl+C to stop)")

    try:
        print("\nPython script running continuously. Waiting for valid sensor data...")
        prev_speed_state = None
        prev_speed_action = None
        
        # State tracking for the Low Pass Filter
        current_left_pwm = 0.0
        current_right_pwm = 0.0
        
        step_count = 0

        while True:
            sensors = client.receive_sensor_data()
            
            if sensors is None or all(v == 0.0 for v in sensors.values()):
                prev_speed_state = None
                prev_speed_action = None
                time.sleep(0.05)
                continue

            # 1. State Mapping
            steer_state = get_state(sensors)
            speed_state = get_speed_state(steer_state)
            
            # 2. Get rewards (Only tracking speed reward now)
            reward = get_speed_reward(speed_state, prev_speed_action if prev_speed_action is not None else 0)
            
            # 3. Update ONLY the Speed Agent
            if training and prev_speed_state is not None:
                speed_agent.update(prev_speed_state, prev_speed_action, reward, speed_state)

            # 4. Choose Actions (Steering is forced to False so it strictly exploits its old table)
            steer_action = choose_action(steering_agent, steer_state, training=False)
            speed_action = choose_speed_action(speed_agent, speed_state, training)
            
            # 5. Apply Speed Multiplier to base steering
            base_left, base_right = ACTIONS[steer_action]
            multiplier = SPEED_MULTIPLIERS[speed_action]
            target_left = base_left * multiplier
            target_right = base_right * multiplier
            
            # 6. Apply the Exponential Moving Average (Low Pass Filter)
            current_left_pwm = (EMA_ALPHA * target_left) + ((1 - EMA_ALPHA) * current_left_pwm)
            current_right_pwm = (EMA_ALPHA * target_right) + ((1 - EMA_ALPHA) * current_right_pwm)
            
            # 7. Bridge communication (Passing raw sensor list to avoid json crash with tuple/string mix)
            raw_sensor_list = [1 if sensors[s]>0.5 else 0 for s in SENSOR_ORDER]
            client.send_motor_command(
                current_left_pwm, current_right_pwm,
                state=raw_sensor_list,  
                reward=reward,
                action=speed_action,
            )

            prev_speed_state, prev_speed_action = speed_state, speed_action
            
            if training:
                step_count += 1
                if step_count % 1000 == 0:
                    save_dual_agents(steering_agent, speed_agent, Q_TABLE_PATH)

            time.sleep(0.05)   # ~20 Hz
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            client.send_motor_command(0.0, 0.0, state=[0,0,0,0,0], reward=0.0, action=0)
        except Exception:
            pass
        client.close()
        if training:
            save_dual_agents(steering_agent, speed_agent, Q_TABLE_PATH)

def main():
    parser = argparse.ArgumentParser(description="Task 1B - Q-Learning")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="train: explore + update Q-table; test: greedy, no update")
    args = parser.parse_args()
    
    run(args.mode)

if __name__ == "__main__":
    main()