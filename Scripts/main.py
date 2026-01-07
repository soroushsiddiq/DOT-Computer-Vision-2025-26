import cv2
import numpy as np
import pyzed.sl as sl
import time
from EllipseFittingFunctions import *
from PoseDeterminationFunctions import *
import csv
import socket
import struct
import os
from collections import deque



## IMPORTANT:
# User must run using python3


# ==========================================================
#                 SYSTEM CONFIGURATION
# ==========================================================

# Enable use of all available CPU cores
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(True)



# ==========================================================
#                 USER PARAMETERS
# ==========================================================

##
##
##
##
##

# ==========================================================
#                 PHYSICAL PARAMETERS
# ==========================================================


# Using this technique, only targeting the inner circle
R_MM = 90.0 #mm
MIN_CONTOUR_AREA = 2000
MIN_CONTOUR_LENGTH = 50
MIN_CONTOUR_POINTS = 50


RANSAC_MAX_TRIALS = 200
CONVERGENCE_TRIALS =90 
RANSAC_INLIER_THRESHOLD = 5
#R_OUTTER = 150.0 # mm, Real radius of the circular lAR
#INNER_OUTTER_OFFSET = 70.0 # mm, distance between inner and outter circular faces



# ==========================================================
#                 CAMERA PARAMETERS
# ==========================================================

FPS = 10
pixel_size_mm_zed = 0.002 # pixel size for ZED at pix/mm

# ==========================================================
#                 DATA WRITING OPTIONS
# ==========================================================

header = [
    "Time [s]",
    "Pose1_Rx", "Pose1_Ry", "Pose1_Rz",
    "Pose1_Tx", "Pose1_Ty", "Pose1_Tz",
    "Pose2_Rx", "Pose2_Ry", "Pose2_Rz",
    "Pose2_Tx", "Pose2_Ty", "Pose2_Tz",
    "True_Rx", "True_Ry", "True_Rz",
    "True_Tx", "True_Ty", "True_Tz"
]

Pose_output = []  # store pose outputs



# ==========================================================
#                 DATA WRITING
# ==========================================================

# Define Base Directory
#CHANGE ME
base_dir = "/home/spot-vision/Documents/GoldSolarPanels/"

# Prompt user for test name
test_name = input("Please enter the test name: ")

print("You entered:", test_name)

# Create the new test directory
new_dir = os.path.join(base_dir, test_name)

# Creates directory if it doesn't exist
os.makedirs(new_dir, exist_ok=True)  

# Build output file paths
csv_output_path = os.path.join(new_dir, f"{test_name}.csv")
vid_path = os.path.join(new_dir, f"{test_name}.mp4")
raw_footage_path = os.path.join(new_dir, f"{test_name}_RAW_FOOTAGE.mp4")

# Print paths for verification
print("CSV Output Path:", csv_output_path)
print("Processed Video Path:", vid_path)
print("Raw Footage Path:", raw_footage_path)


# ==========================================================
#                 WRITING TO SIMULINK
# ==========================================================

# WRITING TO SIMULINK:
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# IP address of groundstation computer

# The second parameter is the port number to send data to, it is arbitrary but must match the Simulink receiver block
server_address = ('192.168.1.110',50005 )



# ==========================================================
#                 HELPER FUNCTIONS
# ==========================================================











# ==========================================================
#                 START ZED CAMERA
# ==========================================================
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Set resolution to HD720
init_params.camera_fps = FPS  # set fps
zed = sl.Camera()
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError("Failed to open ZED camera. Check connection and SDK installation.")

# Get camera intrinsics 
cam_info = zed.get_camera_information()
calib = cam_info.camera_configuration.calibration_parameters
left = calib.left_cam

# Build intrinsic matrix K

K_left = np.array([
    [left.fx, 0.0, left.cx],
    [0.0, left.fy, left.cy],
    [0.0, 0.0, 1.0]
], dtype=float)
print("Obtained left camera intrinsic matrix K:")
print(K_left)

# Setup runtime and Mat for frames
runtime = sl.RuntimeParameters()
frame_zed = sl.Mat()



# ==========================================================
#                 PROCESSING SETUP
# ==========================================================

writer = None
raw_writer = None
frame_count = 0


# grab one frame to get size / FPS for writer
if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
    zed.retrieve_image(frame_zed, sl.VIEW.LEFT)
    # ZED returns BGRA (4-ch); convert to BGR for OpenCV processing
    frame_bgra = frame_zed.get_data()
    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    h, w = frame_bgr.shape[:2] # get frame size
    zed_fps = init_params.camera_fps # get fps
    if vid_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(vid_path, fourcc, zed_fps, (w, h))
        raw_writer = cv2.VideoWriter(raw_footage_path, fourcc, FPS, (w, h))
else:
    zed.close()
    raise RuntimeError("Couldn't read initial frame from ZED.")

print("Running live ZED processing. Press 'q' to quit.")
print(f"Frame size: {w}x{h}, FPS: {zed_fps}")


# ==========================================================
#       MAIN LOOP - Ellipse Detection and Pose Estimation
# ==========================================================

# initialize EPOCH - right before we gegin grabbing frames for ellipse detection
start_time = time.time()
frames = 0


prev_Tx_history = deque(maxlen=5)
try:    
    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            # skip if frame not ready
            if cv2.waitKey(1) & 0xFF == ord('q'): # kill
                break
            continue
        # increment frame count
        current_time = time.time()
        frames += 1

        # Retrieve left image
        zed.retrieve_image(frame_zed, sl.VIEW.LEFT)
        frame_bgra = frame_zed.get_data()


        # Convert to BGR for OpenCV processing
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        # Calculate ellipse from current frame
        ellipse = EllipseFromFrame(frame_bgr)

        if ellipse is not None:
            # Compute two pose candidates based on this ellipse
            try:
                candidates = Ellipse2Pose(R_MM,
                                        K_left,
                                        pixel_size_mm_zed, 
                                        ellipse)
            except:
                candidates=[((0,0,0),(0,0,0)),((0,0,0),(0,0,0))]  # dummy candidates when no ellipse found
            
        else: # if no ellipse, send zeros
           candidates=[((0,0,0),(0,0,0)),((0,0,0),(0,0,0))]  # dummy candidates when no ellipse found           
        
        # Detect mean solar panel slope lines from frame
        mean_slope = detect_panel_lines(frame_bgr)

        # Determine correct pose from direction
        # ---------------------------------------------------
        # Determine sign of Tx based on slope
        # ---------------------------------------------------
        Tx_positive = None
        if ellipse is not None:
            if mean_slope is not None:
                if mean_slope > 0.02:
                    Tx_positive = False
                elif mean_slope < -0.02:
                    Tx_positive = True
                else:
                    Tx_positive = None   # slope too small = ambiguous


        # ---------------------------------------------------
        # Select correct pose candidate
        # ---------------------------------------------------
        if ellipse is not None and Tx_positive is not None:

            Tx1 = candidates[0][1][0]
            Tx2 = candidates[1][1][0]

            Tz1 = candidates[0][1][2]
            Tz2 = candidates[1][1][2]

            #print(Tx1, Tz1, Tx2, Tz2)

            if Tx_positive:
                print("looking RIGHT")
                if Tx1 >= 0 and Tx2 < 0:
                    chosen = candidates[0]
                elif Tx2 >= 0 and Tx1 < 0:
                    chosen = candidates[1]
                else:

                    ##
                    ##
                    # Toggle these two sections: The top halh predicts the next pose based on a linear fit using that last 4 poses
                    # The bottom half predicts the next pose solely based on the last pose - taking the most consistent
                    '''
                    Tx_pred = predict_next_Tx_constant_velocity(prev_Tx_history)
                    print(Tx_pred,Tx1,Tx2)

                    # Compare how close each candidate's Tx is to predicted Tx
                    chosen = candidates[
                        np.argmin([abs(Tx1 - Tx_pred), abs(Tx2 - Tx_pred)])]
                    '''
                    if prev_Tx is not None:
                        chosen = candidates[np.argmin([abs(Tx1 - prev_Tx), abs(Tx2 - prev_Tx)])]
                    else:
                        # if no previous, fall back to smallest |Tx|
                        chosen = candidates[np.argmin([abs(Tx1), abs(Tx2)])]

            else:
                print("looking LEFT")
                if Tx1 <= 0 and Tx2 > 0:
                    chosen = candidates[0]
                elif Tx2 <= 0 and Tx1 > 0:
                    chosen = candidates[1]
                else:
                    ##
                    ##
                    # Toggle these two sections: The top halh predicts the next pose based on a linear fit using that last 4 poses
                    # The bottom half predicts the next pose solely based on the last pose - taking the most consistent
                    '''
                    Tx_pred = predict_next_Tx_constant_velocity(prev_Tx_history)
                    print(Tx_pred,Tx1,Tx2)

                    # Compare how close each candidate's Tx is to predicted Tx
                    chosen = candidates[
                        np.argmin([abs(Tx1 - Tx_pred), abs(Tx2 - Tx_pred)])]
                    '''
                    if prev_Tx is not None:
                        chosen = candidates[np.argmin([abs(Tx1 - prev_Tx), abs(Tx2 - prev_Tx)])]
                    else:
                        # if no previous, fall back to smallest |Tx|
                        chosen = candidates[np.argmin([abs(Tx1), abs(Tx2)])]


        elif ellipse is not None and Tx_positive is None:
            # straight case (slope ambiguous)
            print("Straight")
            Tz1 = candidates[0][1][2]
            Tz2 = candidates[1][1][2]

            # Selects most aligned pose based on which normal has a greater z-component
            chosen = candidates[np.argmax([abs(Tz1), abs(Tz2)])]

        elif is ellipse is None:
            print("No ellipse detected")
            prev_Tx = None

        # ---------------------------------------
        # At the end of each iteration:
        prev_Tx = chosen[1][0]
        prev_Tx_history.append(chosen[1][0])



        Pose_output.append([current_time-start_time, *candidates[0][0], *candidates[0][1], *candidates[1][0], *candidates[1][1], *chosen[0], *chosen[1]])

        # Save raw footage if enabled
        if raw_writer is not None:
            raw_writer.write(frame_bgr)

        # SEND CORRECT POSE CANDIDATE TO SIMULINK:



        try:
            # Send Tx, Tz, Rz for Pose 1

            # Tx: distance between camera optical scenter and LAR center along x in the camera frame (mm)
            # Tz: distance between camera optical center and LAR center along z in the camera frame (mm)
            # Rz: cosine of the angle between camera optical axis and normal to LAR plane around z axis in camera frame

            # To convert to intertial frame of reference, the following transformations are needed:
            # First rotate Tx, Tz to the chaser inertial frame, which is given by the following matrix:
            #| 0  0  1 |
            #| -1 0  0 |

            # Next, GNC must convert from the chaser frame to the intertial lab reference frame, which is given by a principal rotation about the Z axis using the angle of the chaser:
            #| cos(theta) -sin(theta) 0 |
            #| sin(theta)  cos(theta) 0 |

            # Rz = cos(thetaz) -> to get the angle between the camera optical axis and normal to LAR plane around z axis in camera frame, take arccos(Rz)
            # Since camera optical Z axis is aligned with chaser Z axis, the angle between chaser Z axis and normal to LAR plane around z axis in camera frame is also thetaz


            # This is temporary - senging only the first pose for testing
            # Later, integrate with pose disabiguation logic to select correct pose, then send that one.
            Tx = chosen[0][0]# Tx: distance between camera optical scenter and LAR center along x in the camera frame (mm)
            Tz = chosen[0][2] # Tz: distance between camera optical center and LAR center along z in the camera frame (mm)
            Rz = chosen[1][2] # Rz: cosine of the angle between camera optical axis and normal to LAR plane around z axis in camera frame


            data = bytearray(struct.pack("ffff", current_time-start_time, Tx, Tz, Rz))  # Tx, Tz, cos(thetaZ) # Add time later
            sock.sendto(data, server_address)
            
            #print("Data sent!:")
        except Exception as e:
            print(e)
            break


        #print(str(candidates[0][0][0]), str(candidates[0][0][2]), str(candidates[0][1][2]))  
        #print(f" Pose 1 Translation [Tx, Ty, Tz]: {np.round(candidates[0][1], 3)}")
        # Convert threshold image to BGR for drawing & saving (exactly like reference)
        output = frame_bgr

        # Draw ellipse (on threshold output), same color / thickness
        if ellipse is not None:

            cv2.ellipse(output, ellipse, (0, 0, 255), 4)

        # Show window and optionally save
        # Toggle comment on next line to show live window
        #cv2.imshow("RANSAC Circular Fit - ZED Live", output)
        if writer is not None:
            writer.write(output)

except KeyboardInterrupt:
    print("Interrupted by user, stopping...")


# cleanup
if writer is not None:
    writer.release()
if raw_writer is not None:
    raw_writer.release()    
zed.close()
cv2.destroyAllWindows()

# --- Write results to CSV once done ---

with open(csv_output_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(Pose_output)

print(f"\nPose data written to {csv_output_path}")

elapsed = time.time() - start_time
print(f"Frames processed: {frames}, elapsed {elapsed:.2f}s, approx FPS {frames/elapsed:.2f}")
if vid_path:
    print(f"Saved processed video to: {vid_path}")
if raw_footage_path:
    print(f"Saved raw footage to: {raw_footage_path}")
