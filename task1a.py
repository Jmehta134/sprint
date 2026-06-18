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
        control_loop.lpf_error = 0.0  # Added for Low-Pass Filter
        # Start by assuming a White line on a Black background
        control_loop.follow_white_line = True
        control_loop.lost_time = None
        control_loop.lost_cycles = 0  # NEW: Tracks how long we've been lost
        
    left_ext = sensors['left_corner']
    right_ext = sensors['right_corner']

    if left_ext < 0.3 and right_ext < 0.3:
        control_loop.follow_white_line = True
        
    # We expect a White background. If both outer sensors suddenly see Black-> follow white line
    if left_ext > 0.6 and right_ext > 0.6:
        control_loop.follow_white_line = False
        
    # ----- 1. Configuration & Tuning Parameters -----
    Kp = 1.2  # Fixed P: Lower this if it still oscillates wildly
    Ki = 0.0
    Kd = 0.5  # Fixed D: Provides damping. If wobbles are fast/jittery, lower this.
    
    weights = {
        'left_corner': -2.0,
        'left': -1.3,
        'middle': 0.0,
        'right': 1.3,
        'right_corner': 2.0
    }

    # ----- 2. Process Sensors & Check "Lost" Condition -----
    processed_sensors = {}
    for key in SENSOR_ORDER:
        value = sensors[key]
        if control_loop.follow_white_line:
            processed_sensors[key] = value 
        else:
            processed_sensors[key] = 1.0 - value

    # "Lost" Logic: Check if all processed sensor values are within 10% (0.1) of each other
    sensor_values = list(processed_sensors.values())
    max_val = max(sensor_values)
    min_val = min(sensor_values)
    
    if (max_val - min_val) <= 0.1:
        control_loop.lost_cycles += 1
        
        # PENDULUM SEARCH: Prevent spinning 180 degrees and going backward
        spin_speed = 3.5  # Slightly faster to catch the line quickly
        
        # Initial guess based on the last error we saw
        initial_dir = 1 if control_loop.prev_error > 0 else -1
        
        # At 20Hz, ~12 cycles is roughly enough time for a 90-degree turn
        # Sweep 1: 0 to 12 cycles -> Turn up to ~90 deg in the presumed direction
        # Sweep 2: 12 to 36 cycles -> Reverse and sweep ~180 deg to check the other side
        if control_loop.lost_cycles < 5:
            current_dir = initial_dir
        elif control_loop.lost_cycles < 15:
            current_dir = -initial_dir
        else:
            current_dir = initial_dir  # Fallback if really lost
            
        if current_dir > 0:
            return spin_speed, -spin_speed  # Spin Right
        else:
            return -spin_speed, spin_speed  # Spin Left
    else:
        # We found the line! Reset the lost counter immediately
        control_loop.lost_cycles = 0

    # ----- 3. Calculate Line Position Error -----
    numerator = 0.0
    denominator = 0.0
        
    for key in SENSOR_ORDER:
        numerator += weights[key] * processed_sensors[key]
        denominator += processed_sensors[key]

    # ZeroDivisionError Prevention
    if denominator > 1e-6:
        raw_error = numerator / denominator
    else:
        raw_error = control_loop.prev_error  # Fallback to previous error if denominator is 0
    
    # Low-Pass Filter (LPF) applied to the error to smooth sudden spikes
    # Lowered alpha to 0.4 for heavier smoothing of sensor noise
    alpha = 0.4  
    error = (alpha * raw_error) + ((1.0 - alpha) * control_loop.lpf_error)
    control_loop.lpf_error = error

    # ----- 4. PID Math with Deadband -----
    # DEADBAND: If the error is very small, we are essentially on a straight. 
    # Force error to 0 to prevent micro-oscillations (wobbles).
    DEADBAND_THRESHOLD = 0.05
    if abs(error) < DEADBAND_THRESHOLD:
        error = 0.0

    P = Kp * error
    
    control_loop.integral += error
    I = Ki * control_loop.integral
    
    D = Kd * (error - control_loop.prev_error)
    
    turn = P + I + D
    control_loop.prev_error = error

    # ----- 5. Adaptive Speed Logic -----
    MAX_SPEED = 13.0   # Increased from 12.0 (Top straightaway speed)
    MIN_CORNER_SPEED = 2.5  # Increased from 2.5 (Carry more momentum through turns)
    
    K_brake_error = 8.0 # Decreased from 8.0: Brake less aggressively on curves
    K_brake_diff = 4.0  # Decreased from 4.0: Brake less on sudden changes

    # Only brake if we are actually outside the deadband (i.e., entering a real curve)
    if abs(error) > DEADBAND_THRESHOLD:
        adaptive_speed = MAX_SPEED - (K_brake_error * abs(error)) - (K_brake_diff * abs(D))
    else:
        adaptive_speed = MAX_SPEED # Full speed ahead on straights!
    
    # Don't let the base speed drop below your safe cornering speed
    adaptive_speed = max(MIN_CORNER_SPEED, adaptive_speed)

    # ----- 6. Apply to Wheels with Safety Clamps -----
    # Using a slightly negative minimum allows the inner wheel to reverse for tight pivots
    MOTOR_LIMIT_MAX = 13.0 # MUST match or exceed MAX_SPEED (Increased from 12.0)
    MOTOR_LIMIT_MIN = -2.0 # Allowing slightly more reverse power for faster pivots
    
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
