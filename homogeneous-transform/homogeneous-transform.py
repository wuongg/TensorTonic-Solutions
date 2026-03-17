import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    points = np.asarray(points)

    single = False
    if points.ndim == 1:
        points = points[None, :]
        single = True
    ones = np.ones((points.shape[0],1))
    points_h = np.hstack([points,ones])

    transformed_h = (T @ points_h.T).T

    result = transformed_h[:,:3]

    return result[0] if single else result
    pass