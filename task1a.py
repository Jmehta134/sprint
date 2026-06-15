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
        # Start by assuming a White line on a Black background
        control_loop.follow_white_line = True
    
    # Checking and Inverting sensor data only if the background is Black
    threshold = 0.2
    
    left_ext = sensors['left_corner']
    right_ext = sensors['right_corner']
    if control_loop.follow_white_line:
        # We expect a White background. If both outer sensors suddenly see White-> follow black line
        if left_ext < threshold and right_ext < threshold:
            control_loop.follow_white_line = True
    else:
        # We expect a Black background. If both outer sensors suddenly see Black-> follow white line
        if left_ext > threshold and right_ext > threshold:
            control_loop.follow_white_line = False
    
    # ----- 1. Configuration & Tuning Parameters -----
    base_speed = 2
    Kp = 0.8  
    Ki = 0.0
    Kd = 0.0   

    
    weights = {
        'left_corner': -2.0,
        'left': -1.3,
        'middle': 0.0,
        'right': 1.3,
        'right_corner': 2.0
    }

    # ----- 2. Calculate Line Position Error -----
    numerator = 0.0
    denominator = 0.0
    
    for key, value in sensors.items():
        if control_loop.follow_white_line:
            # Black background = ~0.0, White line = ~1.0
            # Keep as is; the white line is already the highest value
            sensors[key] = value 
        else:
            # White background = ~1.0, Black line = ~0.0
            # Invert so the black line becomes the highest value
            sensors[key] = 1.0 - value
        
        numerator += weights[key] * sensors[key]
        denominator += sensors[key]

    # Ensure we actually have a distinct line to follow (denominator isn't effectively 0)
    if denominator > 0.1: 
        error = numerator / denominator
    else:
        # Line is completely lost we continue..
            # error = 0.0
            exit()
    print(error)

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
