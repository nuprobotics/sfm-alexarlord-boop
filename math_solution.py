import numpy as np


def rodrigues_rotation_matrix(rotation_vector):
    theta = np.linalg.norm(rotation_vector)
    if theta < 1e-6:
        return np.eye(3)
    k = rotation_vector / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R


def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
):
    """
    :param camera_matrix: first and second camera matrix, np.ndarray 3x3
    :param camera_position1: first camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation1: first camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param camera_position2: second camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation2: second camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param image_points1: points in the first image, np.ndarray Nx2
    :param image_points2: points in the second image, np.ndarray Nx2
    :return: triangulated points, np.ndarray Nx3
    """

    # we should convert world coordinates to camera coordinates first
    camera_position1 = -camera_rotation1.T @ camera_position1
    camera_position2 = -camera_rotation2.T @ camera_position2

    # projection matrices for the first and second cameras, respectively
    P1 = camera_matrix @ np.hstack((camera_rotation1.T, camera_position1.reshape(-1, 1)))
    P2 = camera_matrix @ np.hstack((camera_rotation2.T, camera_position2.reshape(-1, 1)))

    points_3d = []

    # triangulate each pair of points (linear equation system followed by SVD application) -> 3D points
    for pt1, pt2 in zip(image_points1, image_points2):
        A = np.array([
            pt1[0] * P1[2] - P1[0],
            pt1[1] * P1[2] - P1[1],
            pt2[0] * P2[2] - P2[0],
            pt2[1] * P2[2] - P2[1]
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]
        points_3d.append(X[:3])

    return np.array(points_3d)
