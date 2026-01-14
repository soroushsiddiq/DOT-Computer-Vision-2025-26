import numpy as np


# AUTHOR: JAMES MAKHLOUF
# DATE: 2024-10-15
# DESCRIPTION: Functions for determining the 3D pose of a circle from its elliptical projection in an image.
# The functions implement the algorithm from the papers:
# Miao Xikui et al., “Monocular vision pose measurement based on docking ring component,” Acta Optica Sinica, vol. 33, no. 4, p. 0412006, 2013. doi:10.3788/aos201333.0412006
# and
# J. Mu, S. Li, and M. Xin, “Circular-feature-based pose estimation of noncooperative satellite using time-of-flight sensor,” Journal of Guidance, Control, and Dynamics, vol. 47, no. 5, pp. 840–856, May 2024. doi:10.2514/1.g007629 

# Summary:
# FUNCTION DEFINITIONS:

def normalize_radius(R_mm,k,pixel_size_mm):
    """
    Compute the normalized circle radius (unitless) for use in pose-from-ellipse.
    
    Parameters
    ----------
    R_mm : float
        Radius of the circle in mm
    k : ndarray (3,3)
        Camera intrinsic matrix.
        k = [[fx, s, cx],
             [0, fy, cy],
             [0, 0, 1]]
        fx: focal length in pixels (x-axis)
        fy: focal length in pixels (y-axis)
        s: skew (usually 0)
        (cx, cy): principal point in pixels
    pixel_size_mm : float
        Size of one pixel in mm (pixel pitch).
        Usually on the order of 0.002 mm / pixel
    
    Returns
    -------
    R_px : float
        Normalized (unitless) radius, R_mm / f_mm.
    """
    fx = k[0,0] # focal length in pixels along x-axis
    fy = k[1,1] # focal length in pixels along y-axis
    f_px = (fx + fy) / 2.0 # average focal length in pixels
    
    R_norm = R_mm / (f_px * pixel_size_mm) # convert radius from mm to pixels
    
    return R_norm

def ConicFromEllipse(ellipse_params):
    """
    Convert ellipse parameters to conic coefficients.
    Ellipse parameters from cv2.fitEllipse
    
    Parameters
    ----------
    ellipse_params : tuple
        ((xc, yc), (MA, ma), angle)
        from cv2.fitEllipse
    
    Returns
    -------
    A, B, C, D, E, F : float
        Conic coefficients for Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0

    Test Case:
    Input: center=(1,2), axes=(4,3.5), angle=-27 degrees
    Output: A: 0.06644338087730416
            B: 0.015478641474010468
            C: 0.07768927218392034
            D: -0.16384404470262925
            E: -0.3262357302096918
            F: -0.5918422474389935
    """
    (xc, yc), (MA, ma), angle = ellipse_params
    
    # Semi-axes
    a = MA/2 #semi-major axis (radius)
    b = ma/2 #semi-minor axis (radius)
    
    # Angle in radians (OpenCV gives degrees, convert to radians)
    theta = np.deg2rad(angle)
    
    # Coefficients needed for conic form
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    A = (cos_theta**2)/(a**2) + (sin_theta**2)/(b**2)
    try:
        B = 2*cos_theta*sin_theta*(1/(a**2) - 1/(b**2))
        C = (sin_theta**2)/(a**2) + (cos_theta**2)/(b**2)
    except:
        B = 1000
        C = 1000
    D = -2*A*xc - B*yc
    E = -2*C*yc - B*xc
    F = A*xc**2 + B*xc*yc + C*yc**2 - 1
    

    return A, B, C, D, E, F 

def SymMatrixFromConic(A, B, C, D, E, F):
    """
    Construct the symmetric matrix representation of the conic.
    
    Parameters
    ----------
    A, B, C, D, E, F : float
        Conic coefficients for Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    
    Returns
    -------
    Q : ndarray (3,3)
        Symmetric matrix representing the conic.
    """
    Q_img = np.array([
        [A,   B/2, D/2],
        [B/2, C,   E/2],
        [D/2, E/2, F  ]
    ])
    return Q_img

def normalize_conic(Q_img, K):
    """
    Normalize conic from image coordinates to camera coordinates.
    
    Parameters
    ----------
    Q_img : ndarray (3,3)
        Conic matrix symmetric in image coordinates.
        Q_img = [[A, B/2, D/2],
                 [B/2, C, E/2],
                 [D/2, E/2, F]]
        Where A, B, C, D, E, F are conic coefficients from: 
        Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0

    K : ndarray (3,3)
        Camera intrinsic matrix.
        K = [[fx, s, cx],
             [0, fy, cy],
             [0, 0, 1]]
        fx: focal length in pixels (x-axis)
        fy: focal length in pixels (y-axis)
        s: skew (usually 0)
        (cx, cy): principal point in pixels
    Returns
    -------
    Q_norm : ndarray (3,3)
        Conic matrix in camera coordinates.
        Q_norm = K^T * Q_img * (K)
    """
   
    Q_norm = K.T @ Q_img @ K

    return Q_norm

def eigendecomp(Q):
    """
    Eigen-decomposition of the conic 3x3 matrix.
    Q_norm is an indefinite symmetric matrix since the ellipse projects a real circle.
    Therefore there are three real eigenvalues, two with the same sign and one with the opposite sign.
    lambda_3 is the eigenvalue with the opposite sign to lambda_1 and lambda_2.
    abs(lambda_1) >= abs(lambda_2)
    u3 is the eigenvector associated with lambda_3, and must have a positive z-component (u3[2] > 0). Sign to be flipped if needed.
    Enforce right-handed coordinate system: u1 = u2 x u3

    Geometric Description:
    u3: normal to the plane of the circle in 3D space (unit vector
    u2: one axis in the ellipse's plane
    u1: completees the triad with u1 = u2 x u3

    So, strictly speaking, U is no longer in the eigenbasis since u1 is redefined, but it is still an orthogonal matrix.

    Parameters
    ----------
    Q : ndarray (3,3)
        Symmetric conic matrix.
    
    Returns U, P 
    -------
    U : ndarray (3,3)
        Orthogonal matrix of eigenvectors (columns = eigenvectors).
        U = [u1, u2, u3] where ui are (3x1) eigenvectors.
    P : ndarray (3,3)
        Diagonal matrix of eigenvalues.
        P = diag([λ1, λ2, λ3]) where λi are the eigenvalues.

    """
    eigvals, eigvecs = np.linalg.eigh(Q) #computes eigenvalues and eigenvectors of Q

    eigsigns = [] #list to store signs of eigenvalues
    for val in eigvals:
        eigsigns.append(np.sign(val)) # converts eigvals list to either -1, 0, or 1: -1 for negative, 0 for zero, 1 for positive sign
    
    positives = eigsigns.count(1) #counts number of positive eigenvalues
    negatives = eigsigns.count(-1) #counts number of negative eigenvalues
    if positives == 2 and negatives == 1:
        lambda_3_index = eigsigns.index(-1) #index of the eigenvalue with the opposite sign
    elif positives == 1 and negatives == 2:
        lambda_3_index = eigsigns.index(1) #index of the eigenvalue with the opposite sign
    
    lambda_3 = eigvals[lambda_3_index] #eigenvalue with the opposite sign
    u3 = eigvecs[:, lambda_3_index] #eigenvector associated with lambda_3
    if u3[2] < 0: #ensure u3 has a positive z-component
        u3 = -u3 #flip sign of u3 if needed to ensure positive z-component

    # Remaining eigenpairs
    idxs = [i for i in range(3) if i != lambda_3_index]
    lambda_1, lambda_2 = eigvals[idxs[0]], eigvals[idxs[1]]
    u1, u2 = eigvecs[:, idxs[0]], eigvecs[:, idxs[1]]
    if abs(lambda_1) >= abs(lambda_2):
        pass #lambda_1 and u1 are already correctly assigned
    else:
        #swap lambda_1 and lambda_2, and u1 and u2
        lambda_1, lambda_2 = lambda_2, lambda_1
        u1, u2 = u2, u1 #swap u1 and u2 to maintain correct association with eigenvalues   
    
    #Ensure u2, u3 are unit vectors:
    u2 = u2 / np.linalg.norm(u2)
    u3 = u3 / np.linalg.norm(u3)
    #Enforce right-handed coordinate system, redefine u1:
    u1 = np.cross(u2, u3)
    u1 = u1 / np.linalg.norm(u1)
    
    U = np.column_stack((u1, u2, u3)) #orthogonal matrix
    P = np.diag([lambda_1, lambda_2, lambda_3])
    

    return U, P   

def circle_candidates(U, P, r_norm, k ,pixel_size_mm):
    """
    Compute candidate circle centers and normals in camera coordinates.

    Parameters
    ----------
    U : ndarray (3,3)
        Orthogonal matrix of eigenvectors (columns).
    P : ndarray (3,3)
        Diagonal matrix of eigenvalues.
    r_norm : float
        Normalized circle radius (unitless).
    k : ndarray (3,3)
        Camera intrinsic matrix.
        k = [[fx, s, cx],
             [0, fy, cy],
             [0, 0, 1]]
        fx: focal length in pixels (x-axis)
        fy: focal length in pixels (y-axis)
        s: skew (usually 0)
        (cx, cy): principal point in pixels
    pixel_size_mm : float
        Size of one pixel in mm (pixel pitch).
        Usually on the order of 0.002 mm / pixel
    Returns
    -------
    candidates : list of dict
        Each dict has 'center' and 'normal' in camera coords.
        Candidates = [{'center': c1, 'normal': n1},
                      {'center': c2, 'normal': n2}]
    where c1, c2 are (3x1) circle centers and n1, n2 are (3x1) plane normals corresponding to the two possible poses.
    Each pose is expressed in camera coordinates.
    """
    lam1, lam2, lam3 = np.diag(P)

    # --- circle center candidates in primed frame --- (from the chinese paper)
    xmag = r_norm * np.sqrt(abs(lam3)*(abs(lam1)-abs(lam2))) / np.sqrt( abs(lam1) * ((abs(lam1)+abs(lam3)))) # x-component
    zmag = r_norm * np.sqrt(abs(lam1)*(abs(lam2)+abs(lam3))) / np.sqrt( abs(lam3) * ((abs(lam1)+abs(lam3)))) # z-component

    c1p = np.array([ xmag, 0.0,  zmag]) # first candidate center
    c2p = np.array([-xmag, 0.0, zmag]) # second candidate center

    # --- plane normal in primed frame ---
    
    nx = np.sqrt(abs(lam1)-abs(lam2)) / np.sqrt(abs(lam1)+abs(lam3)) # x-component
    nz = - np.sqrt(abs(lam2)+abs(lam3)) / np.sqrt(abs(lam1)+abs(lam3))  # z-component (negative sign as per paper)

    n1p = np.array([ nx, 0.0, nz]) # first candidate normal
    n2p = np.array([-nx, 0.0, nz]) # second candidate normal
    # --- map back to camera coordinates ---
    c1 = U @ c1p
    n1 = U @ n1p
    c2 = U @ c2p
    n2 = U @ n2p

    # Must scale to mm: normal vectors are unitless directional unit vectors so they need not be scaled,
    # but the centers are in units of normalized focal length, so must be scaled by actual focal length in mm.
    fx = k[0,0] # focal length in pixels along x-axis
    fy = k[1,1] # focal length in pixels along y-axis
    f_px = (fx + fy) / 2.0 # average focal length in pixels
    
    c1 = c1 * f_px * pixel_size_mm # convert center from normalized units to mm
    c2 = c2 * f_px * pixel_size_mm # convert center from normalized units to mm

    return [(c1, n1), (c2, n2)]


def Ellipse2Pose(R_mm, k, pixel_size_mm, ellipse_params):
    """
    Main function to convert ellipse parameters to candidate circle poses (center and normal) in camera coordinates.
    
    Parameters
    ----------
    R_mm : float
        Radius of the circle in mm
    k : ndarray (3,3)
        Camera intrinsic matrix.
        k = [[fx, s, cx],
             [0, fy, cy],
             [0, 0, 1]]
        fx: focal length in pixels (x-axis)
        fy: focal length in pixels (y-axis)
        s: skew (usually 0)
        (cx, cy): principal point in pixels
    pixel_size_mm : float
        Size of one pixel in mm (pixel pitch).
        Usually on the order of 0.002 mm / pixel
    ellipse_params : tuple
        ((xc, yc), (MA, ma), angle)
        from cv2.fitEllipse
    
    Returns
    -------
    candidates : list of dict
        Each dict has 'center' and 'normal' in camera coords.
        Candidates = [{'center': c1, 'normal': n1},
                      {'center': c2, 'normal': n2}]
    where c1, c2 are (3x1) circle centers and n1, n2 are (3x1) plane normals corresponding to the two possible poses.
    Each pose is expressed in camera coordinates.
    """
    # --- Convert ellipse parameters to conic coefficients ---
    A, B, C, D, E, F = ConicFromEllipse(ellipse_params)
    
    # --- Construct symmetric conic matrix ---
    Q_img = SymMatrixFromConic(A, B, C, D, E, F)
    
    # --- Normalize conic to camera coordinates ---
    Q_norm = normalize_conic(Q_img, k)
    
    # --- Eigen-decomposition of normalized conic ---
    U, P = eigendecomp(Q_norm) # U are normalized eigenvectoers, P is diagonal matrix of eigenvalues
    
    # --- Normalize radius ---
    r_norm = normalize_radius(R_mm, k, pixel_size_mm) # normalize radius in pixels
    
    # --- Compute candidate poses ---
    candidates = circle_candidates(U, P, r_norm, k, pixel_size_mm)
    
    return candidates