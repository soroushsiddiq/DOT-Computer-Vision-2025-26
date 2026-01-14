import cv2
import numpy as np
import math
## TAKES IN A BGR FRAME AND THE COMPUTES THE INNER ELLIPSE



#======================================================
#                 USER CONSTANTS

# ==========================================================
# --- RANSAC ellipse fitting parameters ---
RANSAC_MAX_TRIALS = 200
RANSAC_INLIER_THRESHOLD = 7
CONVERGENCE_TRIALS = 90

# CONTOUR FITTING PARAMETERS
MIN_CONTOUR_AREA = 1500

# ELLIPSE QUALITY PARAMETERS:
MAX_AR = 1.4
MIN_INLIER_FRAC = 0.5









#======================================================
#                 HELPER FUNCTIONS:
# ==========================================================


def predict_next_Tx_constant_velocity(prev_Tx_history):
    """
    Predict next Tx assuming constant velocity based on last N points.
    Linear fit: Tx = m*t + b
    """
    n = len(prev_Tx_history)
    if n == 0:
        return 0
    if n == 1:
        return prev_Tx_history[0]  # can't estimate velocity
    
    # t = 1..n
    t = np.arange(1, n+1)
    Tx = np.array(prev_Tx_history)

    # Fit a line
    m, b = np.polyfit(t, Tx, 1)

    # Predict next frame
    t_next = n + 1
    return m * t_next + b


#======================================================
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
    std_dev = np.inf
    no_improvement = 0
    AR = 0
    inlier_frac = 0

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


#======================================================
#                 GOLD MASK FUNCTION
# ==========================================================

def get_gold_mask(frame_bgr,kernel_size=5,iterations=3):
    
    frame_bgr = cv2.GaussianBlur(frame_bgr, (7,7), 0)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(2,2))
    V_eq = clahe.apply(V)
    hsv_eq = cv2.merge([H,S,V_eq])

    lower_gold = np.array([0,85,0])
    upper_gold = np.array([200,255,255])

    mask_lab = cv2.inRange(hsv_eq, lower_gold, upper_gold)

    mask = cv2.morphologyEx(
            mask_lab, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            iterations=iterations
        )
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    
    return mask

#======================================================
#                 SOLAR PANEL LINES
# ==========================================================

def detect_panel_lines(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (13,13),0)
    frame = cv2.medianBlur(frame,3)
    edges = cv2.Canny(frame, 100, 175, apertureSize=3)
    h,w = frame.shape
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=75,
        maxLineGap=10
    )

    slopes = []   # store valid slopes
    border_margin = 20
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if not (
                y1 <= border_margin or y1 >= h - border_margin or
                y2 <= border_margin or y2 >= h - border_margin
            ):

                dx = x2 - x1
                dy = y2 - y1

                # Angle in degrees
                angle = abs(math.degrees(math.atan2(dy, dx)))

                # Reject nearly vertical lines
                if 45 <= angle <= 135:
                    continue
                if -45 >= angle >= -135:
                    continue
                # Compute slope
                if abs(dx) > 1e-6:
                    m = dy / dx
                    slopes.append(m)
    # -----------------------------------
    # Compute MEAN SLOPE of remaining lines
    # -----------------------------------
    slopes = [item for item in slopes if item !=0.0]
    slopes = [s for s in slopes if abs(s) < 0.3]
    if len(slopes) > 0:
        mean_slope = float(np.mean(slopes))
    else:
        mean_slope = None

    return  mean_slope

#======================================================
#                 ELLIPSE FROM FRAME
# ==========================================================
def EllipseFromFrame(frame_bgr):

    gold_mask = get_gold_mask(frame_bgr, kernel_size=5, iterations=3)
    # Currently fitting to gold mask
    gold_contours,_ = cv2.findContours(gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if gold_contours:
        max_contour = max(gold_contours, key=cv2.contourArea)

    else:
        max_contour = None

    
    if max_contour is not None:
        ellipse, best_inliers, best_mse, std_dev, AR, inlier_frac = ransac_fit_ellipse_traditional(max_contour,RANSAC_MAX_TRIALS,CONVERGENCE_TRIALS,RANSAC_INLIER_THRESHOLD)

        if ellipse is not None and (AR < MAX_AR) and inlier_frac > MIN_INLIER_FRAC:
            return ellipse
        else:
            return None
    else:
        return None