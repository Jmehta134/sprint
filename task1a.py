"""
===================================================
    eLSI Sprint 1 - Task 1A : PID Line Following
===================================================

Participant template.

HOW TO RUN
  1. Open the Task 1A scene in CoppeliaSim.
  2. Start the bridge:   python3 bridge_task1a.py --eval
  3. Run this file:      python3 task1a_template.py

WHAT YOU IMPLEMENT
  Only control_loop(). Everything else (connecting, receiving sensors,
  sending motor commands) is handled for you by CoppeliaClient.
  Don't Edit this file except control_loop().
  You can add helper functions if you like.

Team ID: [ XXX ]
"""

import time

from connector_task1a import CoppeliaClient

# The five line sensors, ordered left -> right across the robot.
# Each value is in [0.0, 1.0]; a higher value means the line is detected.
SENSOR_ORDER = ['left_corner', 'left', 'middle', 'right', 'right_corner']


def control_loop(sensors):
    """Return (left_speed, right_speed) to track the lowest sensor reading."""
    
    # ----- 0. Initialize PID State -----
    if not hasattr(control_loop, "prev_error"):
        control_loop.prev_error = 0.0
        control_loop.integral = 0.0

    # ----- 1. Configuration & Tuning Parameters -----
    base_speed = 2
    Kp = 1.5  
    Ki = 0.0
    Kd = 0.0   
    
    weights = {
        'left_corner': 2.0,
        'left': 1.3,
        'middle': 0.0,
        'right': -1.3,
        'right_corner': -2.0
    }

    # ----- 2. Calculate Line Position Error -----
    numerator = 0.0
    denominator = 0.0
    
    for key, value in sensors.items():
        # Subtracting the reading from the max value inverts the data.
        # The lowest sensor reading becomes the highest number (the "mass").
        inverted_val = 1 - value
        
        numerator += weights[key] * inverted_val
        denominator += inverted_val

    # Ensure we actually have a distinct line to follow (denominator isn't effectively 0)
    if denominator > 0.1: 
        error = numerator / denominator
    else:
        # Line is completely lost we continue..
            error = 0.0

    # ----- 3. PID Math -----
    P = Kp * error
    
    control_loop.integral += error
    I = Ki * control_loop.integral
    
    D = Kd * (error - control_loop.prev_error)
    
    turn = P + I + D
    control_loop.prev_error = error

    # ----- 4. Apply to Wheels -----
    left_speed = base_speed + turn
    right_speed = base_speed - turn

    return left_speed, right_speed


def main():
    client = CoppeliaClient(host="127.0.0.1", port=50002)
    client.connect()
    print("Connected to bridge_task1a. Running... (Ctrl+C to stop)")

    last_sensors = None
    try:
        while True:
            # Pull the freshest sensor packet; reuse the last one between packets.
            sensors = client.receive_sensor_data()
            if sensors is not None:
                last_sensors = sensors
            if last_sensors is None:
                time.sleep(0.02)
                continue

            left, right = control_loop (last_sensors)
            client.send_motor_command(left, right)

            time.sleep(0.05)   # ~20 Hz control loop
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            client.send_motor_command(0.0, 0.0)   # stop the robot
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()
