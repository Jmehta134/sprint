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
        control_loop.lost_time = None
        
    left_ext = sensors['left_corner']
    right_ext = sensors['right_corner']

    if left_ext < 0.3 and right_ext < 0.3:
        control_loop.follow_white_line = True

        
    # We expect a White background. If both outer sensors suddenly see Black-> follow white line
    if left_ext > 0.6 and right_ext > 0.6:
        control_loop.follow_white_line = False
        
    # ----- 1. Configuration & Tuning Parameters -----
    base_speed = 2
    Kp = 1.5
    Ki = 0.0
    Kd = 0.4  

    
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
            sensors[key] = value 
        else:
            sensors[key] = 1.0 - value
        
        numerator += weights[key] * sensors[key]
        denominator += sensors[key]

    # calculate error         
    error = numerator / denominator
    
    # ----- 3. PID Math -----
    P = Kp * error
    
    control_loop.integral += error
    I = Ki * control_loop.integral
    
    D = Kd * (error - control_loop.prev_error)
    
    turn = P + I + D
    control_loop.prev_error = error

    # ----- 4. Adaptive Speed Logic -----
    MAX_SPEED = 10.0   # Top speed on straightaways
    MIN_CORNER_SPEED = 2.0  # Slowest base speed allowed so it doesn't stall in turns
    
    K_brake_error = 10.0 # How hard to brake based on current error
    K_brake_diff = 5.0  # How hard to brake when approaching a turn quickly (D-term)

    # Calculate adaptive speed
    adaptive_speed = MAX_SPEED - (K_brake_error * abs(error)) - (K_brake_diff * abs(D))
    
    # Don't let the base speed drop below your safe cornering speed
    adaptive_speed = max(MIN_CORNER_SPEED, adaptive_speed)

    # ----- 5. Apply to Wheels with Safety Clamps -----
    # Using a slightly negative minimum allows the inner wheel to reverse for tight pivots
    MOTOR_LIMIT_MAX = 12.0
    MOTOR_LIMIT_MIN = -2.0 
    
    left_speed = adaptive_speed + turn
    right_speed = adaptive_speed - turn

    # Clamp the final outputs to prevent the simulator from silently freezing!
    left_speed = max(MOTOR_LIMIT_MIN, min(left_speed, MOTOR_LIMIT_MAX))
    right_speed = max(MOTOR_LIMIT_MIN, min(right_speed, MOTOR_LIMIT_MAX))

    return left_speed, right_speed


def main():
    client = CoppeliaClient(host="127.0.0.1", port=50002)
    client.connect()
    print("Connected to bridge_task1a. Running... (Ctrl+C to stop)")

    try:
        while True:
            # Pull the freshest sensor packet; reuse the last one between packets.
            sensors = client.receive_sensor_data()

            if sensors is  None:
                time.sleep(0.01)
                continue
            
            left, right = control_loop (sensors)
            client.send_motor_command(left, right)

            # time.sleep(0.05)   # ~20 Hz control loop
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
