"""
===================================================
  eLSI Sprint 1 - Task 1A : PID Line Following
===================================================

Participant template.

HOW TO RUN
  1. Open the Task 1A scene in CoppeliaSim.
  2. Start the bridge:   python3 bridge_task1a.py --eval
  3. Run this file:      python3 task1a_template.py

Team ID: [ 782 ]
"""

import time
from connector_task1a import CoppeliaClient

SENSOR_ORDER = ['left_corner', 'left', 'middle', 'right', 'right_corner']
WEIGHTS = {'left_corner': -2.5, 'left': -1.5, 'middle': 0.0, 'right': 1.5, 'right_corner': 2.5}

def control_loop(sensors, _state=[0.0, 0]):
    """
    Ultra-fast PD control loop for line following.
    
    How it works:
    1. Speed & State: Uses a default mutable list `_state` to persist `prev_error` 
       and `lost_cycles` across frames without the latency of global variables or object attributes.
    2. Contrast Detection: Calculates track contrast (max_val - min_val). If contrast 
       is <= 0.1, the robot is reading a solid color (lost state) and snaps into a hard turn.
    3. Auto-Inversion: Checks the outer sensors to determine if the track is dark-on-light 
       or light-on-dark, dynamically inverting the readings if necessary.
    4. Algebraic Optimization: The P and D terms are factored together algebraically 
       (turn = 2.0 * error - 0.8 * prev_error) to eliminate unnecessary math operations.
       
    State array map:
    _state[0] = prev_error
    _state[1] = lost_cycles
    """
    
    # 1. Direct variable assignment (bypassing slow dictionary lookups)
    lc = sensors['left_corner']
    l  = sensors['left']
    m  = sensors['middle']
    r  = sensors['right']
    rc = sensors['right_corner']

    # 2. Lost State Detection (Contrast-based)
    max_val = max(lc, l, m, r, rc)
    min_val = min(lc, l, m, r, rc)

    if (max_val - min_val) <= 0.1:
        # --- LOST STATE ---
        _state[1] += 1
        
        # Hard turn in the last known direction (2.5 is the max sensor weight)
        error = 3 if _state[0] > 0 else -3
    else:
        # --- FOUND STATE ---
        _state[1] = 0
        
        # Auto-invert values if the background is dark (edges are triggering highly)
        if lc > 0.6 and rc > 0.6:
            lc, l, m, r, rc = 1.0 - lc, 1.0 - l, 1.0 - m, 1.0 - r, 1.0 - rc
            
        den = lc + l + m + r + rc
        
        # Calculate weighted error (middle weight is 0.0, so it is omitted from the numerator)
        if den > 0.001:
            error = (-2.5 * lc - 1.5 * l + 1.5 * r + 2.5 * rc) / den
        else:
            error = _state[0]

    # 3. Pre-calculated PD Math (Equivalent to Kp=1.2, Kd=0.8)
    turn = 1.0 * error - 0.2 * _state[0]
    _state[0] = error

    # 4. Inline Absolute Value & Adaptive Speed
    abs_err = error if error > 0 else -error
    speed = 12.0 - (2.8  * abs_err)

    left_speed = speed + turn
    right_speed = speed - turn

    # 5. Inline Clamping (-2.0 to 14.0 bounds)
    left_speed = 14.0 if left_speed > 14.0 else (-2.0 if left_speed < -2.0 else left_speed)
    right_speed = 14.0 if right_speed > 14.0 else (-2.0 if right_speed < -2.0 else right_speed)

    return left_speed, right_speed

def main():
    client = CoppeliaClient(host="127.0.0.1", port=50002)
    client.connect()
    print("Connected to bridge_task1a. Running... (Ctrl+C to stop)")

    try:
        while True:
            sensors = client.receive_sensor_data()
            if sensors is None:
                continue
            
            left, right = control_loop(sensors)
            client.send_motor_command(left, right)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            client.send_motor_command(0.0, 0.0)
        except Exception:
            pass
        client.close()

if __name__ == "__main__":
    main()