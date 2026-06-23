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

Team ID: [ 782 ]
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
        control_loop.lpf_error = 0.0  
        control_loop.follow_white_line = True
        control_loop.lost_time = None
        control_loop.lost_cycles = 0  
        
    left_ext = sensors['left_corner']
    right_ext = sensors['right_corner']

    if left_ext < 0.3 and right_ext < 0.3:
        control_loop.follow_white_line = True
        
    if left_ext > 0.6 and right_ext > 0.6:
        control_loop.follow_white_line = False
        
    # ----- 1. Configuration & Tuning Parameters -----
    Kp = 0.8  
    Ki = 0.0
    Kd = 0.5  
    
    weights = {
        'left_corner': -2.5,
        'left': -1.6,
        'middle': 0.0,
        'right': 1.6,
        'right_corner': 2.5
    }

    # ----- 2. Process Sensors -----
    processed_sensors = {}
    for key in SENSOR_ORDER:
        value = sensors[key]
        if control_loop.follow_white_line:
            processed_sensors[key] = value 
        else:
            processed_sensors[key] = 1.0 - value

    # ----- 3. "Lost" Logic & Error Calculation -----
    sensor_values = list(processed_sensors.values())
    max_val = max(sensor_values)
    min_val = min(sensor_values)

    if (max_val - min_val) <= 0.1:
        # --- LOST STATE ---
        control_loop.lost_cycles += 1
        
        # 1. Determine direction based on the last known error.
        # Since prev_error is updated at the end of every loop, it retains the 
        # sign of the direction we were turning exactly when we lost the line.
        direction = 1.0 if control_loop.prev_error >= 0 else -1.0
        
        # 2. Assume a threshold error, continuously increasing.
        # Start at 1.5 (stronger than the outer sensor weight of 1.0)
        BASE_THRESHOLD = 1.5 
        GROWTH_PER_CYCLE = 0.02  
        
        # Inject the growing error directly into the PID flow
        error = direction * (BASE_THRESHOLD + (GROWTH_PER_CYCLE * control_loop.lost_cycles))
        
        # Keep the LPF synced so it doesn't violently snap when the line is found again
        control_loop.lpf_error = error
        
    else:
        # --- FOUND STATE ---
        control_loop.lost_cycles = 0
        
        numerator = 0.0
        denominator = 0.0
            
        for key in SENSOR_ORDER:
            numerator += weights[key] * processed_sensors[key]
            denominator += processed_sensors[key]

        # ZeroDivisionError Prevention
        if denominator > 1e-6:
            raw_error = numerator / denominator
        else:
            raw_error = control_loop.prev_error  
        
        # Low-Pass Filter (LPF) applied to the error
        alpha = 0.6  
        error = (alpha * raw_error) + ((1.0 - alpha) * control_loop.lpf_error)
        control_loop.lpf_error = error

    # ----- 4. PID Math -----
    P = Kp * error
    
    control_loop.integral += error
    I = Ki * control_loop.integral
    
    D = Kd * (error - control_loop.prev_error)
    
    turn = P + I + D
    control_loop.prev_error = error

    # ----- 5. Adaptive Speed Logic -----
    MAX_SPEED = 14.0   
    MIN_CORNER_SPEED = 2.5  
    
    K_brake_error = 10.0 
    K_brake_diff = 8.0  
    
    # Restored the adaptive speed calculation so it doesn't throw a reference error
    adaptive_speed = MAX_SPEED - (K_brake_error * abs(error)) - (K_brake_diff * abs(D))
    
    # Don't let the base speed drop below your safe cornering speed
    adaptive_speed = max(MIN_CORNER_SPEED, adaptive_speed)

    # ----- 6. Apply to Wheels with Safety Clamps -----
    MOTOR_LIMIT_MAX = 14.0 
    MOTOR_LIMIT_MIN = -2.0 
    
    left_speed = adaptive_speed + turn
    right_speed = adaptive_speed - turn

    # Clamp the final outputs
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