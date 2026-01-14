# ==========================================================
#  AUTONOMOUS SPACE ROBOTICS (DOT) 2025 - 2026
#  Computer Vision - Image Pre-Processing for Ellipse Detection
#  By: Soroush Siddiq (101226772), James Makhlouf (101224410)
#  
#  Description:
#  This script preprocesses video frames to enhance ellipse detection.
#  It includes noise reduction, contrast enhancement, thresholding,
#  contour extraction, filtering, and RANSAC ellipse fitting.
#  The processed frames can be visualized and saved to an output video.
#  User-configurable parameters allow tuning of each processing step.
#  Dependencies: OpenCV, NumPy
#  Last Updated: Nov 2025
# ==========================================================

import cv2
import numpy as np
#from Pose_Determination_Functions import  *
import csv
import matplotlib.pyplot as plt
from collections import deque

# Initialize before your processing loop
prev_candidates = deque(maxlen=10)  # keeps only the last 5


# ==========================================================
#                 USER CONFIGURATION
# ==========================================================
R_outter = 150.0*1.03 # mm, Real radius of the circular lAR
R_inner = 90.0 #mm
INNER_OUTTER_OFFSET = 70.0  # mm, distance between inner and outter circular faces


header = [
    "Time [s]",
    "Pose1_Rx", "Pose1_Ry", "Pose1_Rz",
    "Pose1_Tx", "Pose1_Ty", "Pose1_Tz",
    "Pose2_Rx", "Pose2_Ry", "Pose2_Rz",
    "Pose2_Tx", "Pose2_Ty", "Pose2_Tz"
]

#camera matrix, k
k = np.array([[684.47, 0, 621.47],
              [0, 684.47, 360.28],
              [0, 0, 1]])  # Example intrinsic matrix 

scale = 1
pixel_size_mm = 0.002   # mm/pixel
Pose_output = []  # store pose outputs


# --- Contour filtering parameters ---
MIN_CONTOUR_AREA = 5000
MIN_CONTOUR_LENGTH = 100
MIN_CONTOUR_POINTS = 150


# --- RANSAC ellipse fitting parameters ---
RANSAC_MAX_TRIALS = 200
RANSAC_INLIER_THRESHOLD = 5
CONVERGENCE_TRIALS = 90



# --- Video processing parameters ---

DISPLAY_SCALE = 0.7


# ==========================================================
#                 HELPER FUNCTIONS
# ==========================================================

def get_Z(candidate):
    """Extract Z value safely from tuple/list structures."""
    if candidate is not None:
        return candidate[0][2]
    else:
        return 0


def contour_properties(cnt):
    M = cv2.moments(cnt)
    area = M['m00']
    if area == 0:
        return None, None, None
    cx = M['m10'] / area
    cy = M['m01'] / area
    centroid = np.array([cx, cy])
    return area, centroid, M

def contour_similarity(cnt1, cnt2, centroid_weight=1.0, area_weight=5.0):
    A1, C1, _ = contour_properties(cnt1)
    A2, C2, _ = contour_properties(cnt2)
    if A1 is None or A2 is None:
        return np.inf  # invalid contours

    # Area similarity ratio (close to 1 if similar)
    area_ratio = min(A1, A2) / max(A1, A2)

    # Centroid distance (pixels)
    centroid_dist = np.linalg.norm(C1 - C2)

    # Combine into a similarity score
    score = centroid_weight * centroid_dist + area_weight * abs(1 - area_ratio)
    return score, area_ratio, centroid_dist




# ==========================================================
#                 Angle Variance within contour function
# ==========================================================
def contour_smoothness(contour, window=10):
    contour = contour.squeeze()
    dx = np.gradient(contour[:, 0])
    dy = np.gradient(contour[:, 1])
    angles = np.arctan2(dy, dx)
    dtheta = np.diff(angles)
    dtheta = np.unwrap(dtheta)
    smoothness = np.var(dtheta)  # variance of angle change
    return smoothness

# ==========================================================
#                 Traditional Ransac FUNCTION
# ==========================================================

def ransac_fit_ellipse_traditional(points, max_trials, convergence_trials, inlier_threshold, frame_height=None):
    import cv2, numpy as np

    points = np.asarray(points).reshape(-1, 2)
    if len(points) < 5:
        return None, None, None, None, None,None

    best_inliers = None
    best_ellipse = None
    best_score = -np.inf
    best_mse = np.inf
    no_improvement = 0

    for _ in range(max_trials):
        # --- Random sample fit ---
        try:
            sample = points[np.random.choice(len(points), 5, replace=False)]
            ellipse = cv2.fitEllipse(sample)
        except:
            continue

        (cx, cy), (major_axis, minor_axis), angle_deg = ellipse
        a, b = major_axis / 2, minor_axis / 2
        if a < 1 or b < 1:
            continue

        # --- Residuals for all points ---
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x, y = points[:, 0] - cx, points[:, 1] - cy
        xr = cos_a * x + sin_a * y
        yr = -sin_a * x + cos_a * y
        residuals = (xr / a) ** 2 + (yr / b) ** 2 - 1
        abs_residuals = np.abs(residuals)

        # --- Inlier counting ---
        inliers = abs_residuals < (inlier_threshold / max(a, b))
        num_inliers = np.sum(inliers)

        # --- MSE across *all* points ---
        mse = np.mean(residuals ** 2)

        # --- Model scoring ---
        if num_inliers > best_score or (num_inliers == best_score and mse < best_mse):
            best_score = num_inliers
            best_inliers = inliers
            best_ellipse = ellipse
            best_mse = mse
            no_improvement = 0
        else:
            no_improvement += 1

        # --- Convergence early stop ---
        if no_improvement > convergence_trials:
            break

    # --- Optional refinement using inliers ---
    if best_ellipse is not None and np.sum(best_inliers) >= 5:
        try:
            refined = cv2.fitEllipse(points[best_inliers])
            ellipse_to_use = refined
        except:
            ellipse_to_use = best_ellipse
    else:
        ellipse_to_use = best_ellipse

    # --- Recalculate global fit quality ---
    if ellipse_to_use is not None:
        (cx, cy), (major_axis, minor_axis), angle_deg = ellipse_to_use
        a, b = major_axis / 2, minor_axis / 2
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x, y = points[:, 0] - cx, points[:, 1] - cy
        xr = cos_a * x + sin_a * y
        yr = -sin_a * x + cos_a * y
        residuals = (xr / a) ** 2 + (yr / b) ** 2 - 1
        best_mse = np.mean(residuals ** 2)
        std_dev = np.std(residuals)
        AR = max(a,b) / (min(a,b) +0.01)
        inlier_frac = len(points[best_inliers]) / len(points)
    if best_mse > .5:
        ellipse_to_use = None
    return ellipse_to_use, points[best_inliers], best_mse, std_dev, AR, inlier_frac

# ==========================================================
#                 Gold MLI Mask Function
# ==========================================================
def get_gold_mask(frame_bgr,kernel_size=5,iterations=3):
    
    frame_bgr = cv2.GaussianBlur(frame_bgr, (7,7), 0)
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(frame_hsv)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(2,2))
    V_eq= clahe.apply(V)
    h_blurred = cv2.GaussianBlur(H, (9,9), 0)
    hsv_eq = cv2.merge([h_blurred,S,V_eq])

    lower_gold = np.array([0,7,111])
    upper_gold = np.array([40,255,255])

    mask_lab = cv2.inRange(hsv_eq, lower_gold, upper_gold)

    mask = cv2.morphologyEx(
            mask_lab, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            iterations=iterations
        )
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    return mask



# ==========================================================
#                 MAIN VIDEO PROCESSING FUNCTION
# ==========================================================

def preprocess_video(video_path, out_path,output_csv, display=True):
    """
    Process a video for ellipse detection
    Producs a grid of intermediate steps for debugging
    Returns a csv with pose data from Pose candidate 1.
    """

    # --- Load video ---
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Could not open video."

    # Get fps, width, height from video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- Optional output writer ---
    # Example before the frame loop
    ret, test_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame from video.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    # Generate first combined_frame to know its true dimensions
    resized_bgr = cv2.resize(test_frame, (test_frame.shape[1] // 2, test_frame.shape[0] // 2), interpolation=cv2.INTER_AREA)
    gray_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
    test_img = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    row1 = np.hstack([test_img, test_img, test_img])
    row2 = np.hstack([test_img, test_img, test_img])
    combined_test = np.vstack([row1, row2])

    # --- Define output video size dynamically ---
    frame_h, frame_w = combined_test.shape[:2]
    print(f"Output video size: {frame_w}x{frame_h}")


    # --- Initialize VideoWriter ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))

    if not video_writer.isOpened():
        raise RuntimeError(" VideoWriter failed to open. Check codec or output path.")
    else:
        print(f" VideoWriter initialized successfully at {out_path}")

    # INITIALIZE VARIABLES FOR PROCESSING
    frame = 0   # inital frame
    
    best_inliers = None # empty list of best_inliers
    similarities = []

    # ==========================================================
    #                 VIDEO PROCESSING LOOP
    # ==========================================================
    
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame += 1
        # --------------------------------------------------
        # 1. Gold Mask
        # --------------------------------------------------
        gold_mask = get_gold_mask(frame_bgr, kernel_size=7, iterations=3)

        # --------------------------------------------------
        # 2. Convert to grayscale
        # --------------------------------------------------
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Apply gold mask to grayscale image
        gray_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=cv2.bitwise_not(gold_mask))


        # --------------------------------------------------
        # 3. Noise reduction
        # --------------------------------------------------
        blur_gauss = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        blur_med = cv2.medianBlur(blur_gauss, 7)

       
        # --------------------------------------------------
        # 4. Contrast enhancement (CLAHE)
        # --------------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(3, 3))
        contrast_img = clahe.apply(blur_med)

        # --------------------------------------------------
        # 5. Highlight suppression
        # --------------------------------------------------
        clipped_img = np.clip(contrast_img, 0, 240).astype(np.uint8)

        # --------------------------------------------------
        # 6. Adaptive thresholding
        # --------------------------------------------------
        
        otsu_mask = cv2.threshold(clipped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        clipped_img = cv2.bitwise_and(clipped_img, clipped_img, mask=otsu_mask)
        clipped_img - cv2.GaussianBlur(clipped_img, (5,5),0)
        adaptive_mask = cv2.adaptiveThreshold(
            clipped_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
        9,10
        )

        # --------------------------------------------------
        # 7. Morphological cleanup
        # --------------------------------------------------
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph_clean = cv2.morphologyEx(adaptive_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=2)

        # CONTOUR FITTING
        contours, _ = cv2.findContours(morph_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Filter out smaller contours, keep only large contours
        # The goal of this is to select the LAR outter Contour
        large_contours = [
            c for c in contours
            if cv2.contourArea(c) > MIN_CONTOUR_AREA
            and cv2.arcLength(c, True) > MIN_CONTOUR_LENGTH
            and c.shape[0] > MIN_CONTOUR_POINTS
        ]
        # Return a tree of contours - this will caputre the inner LAR circle
        contours2, hierarchy = cv2.findContours(morph_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if hierarchy is not None:
            hierarchy = hierarchy[0]  # shape (N, 4)

            # Apply the same size filters, but keep indices
            valid_indices = [
                i for i, c in enumerate(contours2)
                if cv2.contourArea(c) > MIN_CONTOUR_AREA
                and cv2.arcLength(c, True) > MIN_CONTOUR_LENGTH
                and c.shape[0] > MIN_CONTOUR_POINTS
            ]

            if not valid_indices:
                most_nested_contour = None
            else:
                # Compute nesting depth for only valid contours
                depths = []
                for i in valid_indices:
                    depth = 0
                    parent = hierarchy[i][3]
                    while parent != -1:
                        depth += 1
                        parent = hierarchy[parent][3]
                    depths.append(depth)

                # Find contour with maximum nesting depth among valid ones
                max_depth_idx = valid_indices[int
                (np.argmax(depths))]

                # the most nested filtered contour corresponds to the inner LAR circle
                most_nested_contour = contours2[max_depth_idx]

        else:
            most_nested_contour = None

        
        # --------------------------------------------------
        # 10. Contour Cleaning and Selection
        # --------------------------------------------------

        # initially, operate with the pretense that we are able to fit an ellipse to outter LAR

        # Choose the contour with the most area:
        if large_contours:
            sorted_contours = sorted(large_contours, key=lambda c: cv2.contourArea(c), reverse=True)
            largest_contour = sorted_contours[0]
            external_contour = largest_contour
        else:
            external_contour = None
       
        # --------------------------------------------------
        # 11. Ellipse fitting
        # --------------------------------------------------
        try:
            ellipse, best_inliers, best_mse, std_dev, AR, inlier_frac = ransac_fit_ellipse_traditional(
                    external_contour,
                    max_trials=RANSAC_MAX_TRIALS,
                    convergence_trials=CONVERGENCE_TRIALS,
                    inlier_threshold=RANSAC_INLIER_THRESHOLD)
        except:
            ellipse , best_inliers, best_mse, std_dev, AR, inlier_frac= None, None, None, None, None, None
        if ellipse is not None:
            '''
            try:
                candidates = Ellipse2Pose(R_outter,k,pixel_size_mm,ellipse)
            except:
                candidates = None
            '''
        # --------------------------------------------------
        # 12. Visualization
        # --------------------------------------------------
        display_otsu_mask = cv2.cvtColor(clipped_img, cv2.COLOR_GRAY2BGR)
        display_best_contour = cv2.cvtColor(morph_clean, cv2.COLOR_GRAY2BGR)
        display_most_nested = cv2.cvtColor(morph_clean, cv2.COLOR_GRAY2BGR)
        display_MLI_mask= cv2.cvtColor(gold_mask,cv2.COLOR_GRAY2BGR)
        display_inliers = frame_bgr.copy()
        display_ellipse = frame_bgr.copy()

        # Draw most nested contour
        if most_nested_contour is not None:
            for pt in most_nested_contour:
                color = (50,255,0)
                x, y = pt[0]
                cv2.circle(display_most_nested, (x, y), 4, color, -1)

        # Draw best inlier points
    
        if best_inliers is not None:
            
            for (x,y) in best_inliers.astype(int):
                color = (255,0,0)#blue
                cv2.circle(display_inliers,(x,y),5,color,-1)

        # Draw best contour points
        # Generate consistent colors for each contour based on its order in the sorted list
        
        for i, contour in enumerate(large_contours):
            color = (50+i * 50 % 256, 89+i * 100 % 256, 150- i * 150 % 256)  # Generate a color based on the index
            for pt in contour:
                x, y = pt[0]
                cv2.circle(display_best_contour, (x, y), 4, color, -1)
        
        if external_contour is not None:
            for pt in external_contour:
                color = (23,64,190)
                x,y = pt[0]
                cv2.circle(display_best_contour,(x,y),4,color,-1)
        fit_contour = external_contour
        if fit_contour is not None:
            for pt in fit_contour:
                color = (255,64,190)
                x,y = pt[0]
                cv2.circle(display_best_contour,(x,y),4,color,-1)
        # Draw ellipse fit
        if ellipse is not None:
                cv2.ellipse(display_ellipse, ellipse, (0, 0, 255), 4)
                #print(candidates[0])
               
         
        # Helper function for titles
        def add_title(img, title):
            img = img.copy()
            cv2.putText(
                img, title, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA
            )
            return img

        
        
        # Add titles to each image
        display_otsu_mask = add_title(display_otsu_mask, "Otsu Mask")
        display_most_nested = add_title(display_most_nested, "Most Nested Contour")
        display_best_contour = add_title(display_best_contour, "External Contour")
        display_MLI_mask = add_title(display_MLI_mask, "Gold MLI Mask")
        display_inliers = add_title(display_inliers, "Best Inliers")
        display_ellipse = add_title(display_ellipse, "Ellipse Fit")


        # --- Ensure all display images have identical dimensions ---
        h_min = min(display_MLI_mask.shape[0], display_inliers.shape[0], display_ellipse.shape[0])
        w_min = min(display_MLI_mask.shape[1], display_inliers.shape[1], display_ellipse.shape[1])

        def resize_to_min(img):
            return cv2.resize(img, (w_min, h_min), interpolation=cv2.INTER_AREA)

        display_MLI_mask = resize_to_min(display_MLI_mask)
        display_inliers  = resize_to_min(display_inliers)
        display_ellipse  = resize_to_min(display_ellipse)

        # Combine all into one visualization grid
        row1 = np.hstack([display_otsu_mask, display_most_nested, display_best_contour])
        row2 = np.hstack([display_MLI_mask, display_inliers, display_ellipse])
        combined_frame = np.vstack([row1, row2])

        if video_writer:
            video_writer.write(combined_frame)

        if display:
            small_display = cv2.resize(combined_frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.imshow("Post-Processed Ellipse Fit", small_display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    

    with open(output_csv, mode="w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        csv_writer.writerows(Pose_output)

    # --- Cleanup ---
    cap.release()
    if video_writer:   # <- use your actual VideoWriter variable, not the CSV writer
        video_writer.release()
    if display:
        cv2.destroyAllWindows()
    
   

# ==========================================================
#                     SCRIPT ENTRY POINT
# ==========================================================

if __name__ == "__main__":
   
    # Further away paths:
    video = "/Users/jamesmakhlouf/Desktop/UNIVERSITY/YEAR 4/Fall 2025/MAAE 4907/MAAE 4907 Q/Test Datasets/Nov 4 Orbit Test 85 cm Headlamp ON/Orbit_Test_NOV4Test_RAWFOOTAGE.mp4"
    out_path = "/Users/jamesmakhlouf/Desktop/UNIVERSITY/YEAR 4/Fall 2025/MAAE 4907/MAAE 4907 Q/Test Datasets/Nov 4 Orbit Test 85 cm Headlamp ON/Post Processed/PoseDisamb-Jan12Test.mp4"
    output_csv = "/Users/jamesmakhlouf/Desktop/UNIVERSITY/YEAR 4/Fall 2025/MAAE 4907/MAAE 4907 Q/Test Datasets/Nov 4 Orbit Test 85 cm Headlamp ON/Post Processed/PoseDisamb-Jan12Test.csv"

    

    preprocess_video(video, out_path, output_csv, display=True)
