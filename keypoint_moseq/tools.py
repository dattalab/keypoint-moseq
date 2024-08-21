"""
General utilities for doing science with keypoint data and/or syllable sequences.
"""

import numpy as np
import annoy
from scipy.spatial import ConvexHull, KDTree
from shapely.geometry import Polygon, Point

# simpleicp
# shapely


def find_arena(
    coordinates,
    arena_shape="auto",
    num_samples=10000,
    k=20,
    distance_threshold=2,
    tolerance=1e-5,
    max_iterations=100,
):
    """
    Detect arena position based on a sample of keypoint coordinates.

    The caller can either provide a target arena shape, which will then be scaled,
    rotated, and translated to best fit the convex hull of the coordinates, or
    the convex hull can be returned directly. Note that this function only works for
    convex arenas.

    Prior to finding the convex hull, the coordinates are randomly sampled and outliers
    are removed using a density estimation (see `detect_outliers` for details). The
    iterative closest point algorithm is then used to find the best transformation
    that aligns the target arena shape to the convex hull (see `icp` for details)

    Parameters
    ----------
    coordinates : np.ndarray
        Array of shape (..., 2) containing 2D keypoint coordinates.

    arena_shape : np.ndarray | str
        Shape of the arena to detect. Can be a (N, 2) array of points defining a polygon,
        or one of the following strings: 'circle', 'square', 'auto'. If 'auto', the
        convex hull of the coordinates will be returned.

    num_samples : int
        Number of samples from `coordinates` to use for arena detection.

    k : int, default=20
        Number of neighbors to use for outlier detection (see `remove_outliers`).

    distance_threshold : float, default=2
        Threshold used for outlier detection (see `remove_outliers` for details).

    tolerance : float, default=1e-5
        Tolerance for the iterative closest point algorithm.

    max_iterations : int, default=100
        Maximum number of iterations for the iterative closest point algorithm.

    Returns
    -------
    arena : np.ndarray
        Array of shape (N, 2) representing the arena as a polygon.
    """
    assert coordinates.shape[-1] == 2, "Coordinates must be 2D"
    coordinates = coordinates.reshape(-1, 2)

    # Randomly sample coordinates
    if num_samples < coordinates.shape[0]:
        idx = np.random.choice(coordinates.shape[0], num_samples, replace=False)
        coordinates = coordinates[idx]

    # Remove outliers
    inlier_mask = detect_outliers(coordinates, k, distance_threshold)
    denoised_coords = coordinates[inlier_mask]

    # Find convex hull
    hull = denoised_coords[ConvexHull(denoised_coords).vertices]

    if arena_shape == "auto":
        arena = hull  # TODO: cleanup_hull(hull)
    else:
        if isinstance(arena_shape, np.ndarray):
            target_polygon = arena_shape
        elif arena_shape == "circle":
            target_polygon = np.array(
                [
                    np.cos(np.linspace(0, 2 * np.pi, 100)),
                    np.sin(np.linspace(0, 2 * np.pi, 100)),
                ]
            ).T
        elif arena_shape == "square":
            target_polygon = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
        else:
            raise ValueError(
                "Invalid arena_shape. Must be 'auto', 'circle', 'square', an array of points."
            )

        arena = find_optimal_transformation(
            target_polygon, hull, tolerance, max_iterations
        )

    return arena


def detect_outliers(coordinates, k=20, distance_threshold=2):
    """
    Detects outliers using the k-nearest neighbors density estimation.

    Parameters
    ----------
    coordinates : np.ndarray
        Array of shape (..., 2) containing 2D keypoint coordinates.

    k : int, default=20
        Number of neighbors to use for outlier detection.

    distance_threshold : float, default=2
        Threshold used for outlier detection. Points whose average distance to their
        k-nearest neighbors is greater than `distance_threshold` times the median
        across all points are considered outliers.

    Returns
    -------
    inlier_mask : np.ndarray
        Boolean mask indicating whether each point is an inlier.
    """
    index = annoy.AnnoyIndex(coordinates.shape[1], "euclidean")
    for i, point in enumerate(coordinates):
        index.add_item(i, point)

    index.build(10)  # 10 trees for improved accuracy

    average_distances = []
    for i in range(len(coordinates)):
        _, distances = index.get_nns_by_item(i, k, include_distances=True)
        average_distances.append(np.mean(distances))

    threshold = distance_threshold * np.median(average_distances)
    inlier_mask = np.array(average_distances) < threshold
    return inlier_mask


def find_optimal_transformation(
    target_polygon, hull, tolerance=1e-5, max_iterations=100
):
    """
    Finds the rotation, translation, and scaling that best aligns the target polygon to
    the convex hull. The scaling factor is determined by ratio of enclosed areas and the
    translation and rotation are determined using the iterative closest point algorithm.

    Parameters
    ----------
    target_polygon : np.ndarray
        Array of shape (N, 2) representing the target arena shape.

    hull : np.ndarray
        Array of shape (M, 2) representing a convex hull.

    tolerance : float, default=1e-5
        Tolerance for the iterative closest point algorithm.

    max_iterations : int, default=100
        Maximum number of iterations for the iterative closest point algorithm.

    Returns
    -------
    transformed_polygon : np.ndarray
        Array of shape (N, 2) representing the transformed target polygon.
    """
    # Rescale based on area
    hull_area = Polygon(hull).area
    target_area = Polygon(target_polygon).area
    target_polygon = target_polygon * np.sqrt(hull_area / target_area)

    # Perform ICP
    for i in range(max_iterations):
        closest, distances = _closest_points(target_polygon, hull)
        R, t = _estimate_transformation(target_polygon, closest)
        target_polygon = np.dot(target_polygon, R.T) + t

        # Check for convergence
        if np.mean(distances) < tolerance:
            break

    return target_polygon


def _closest_points(source, target):
    target_kd_tree = KDTree(target)
    distances, indices = target_kd_tree.query(source)
    return target[indices], distances


def _estimate_transformation(source, closest):
    # Compute centroids
    centroid_source = np.mean(source, axis=0)
    centroid_closest = np.mean(closest, axis=0)

    # Subtract centroids
    source_centered = source - centroid_source
    closest_centered = closest - centroid_closest

    # Singular Value Decomposition (SVD) for rotation
    U, _, VT = np.linalg.svd(np.dot(source_centered.T, closest_centered))
    R = np.dot(VT.T, U.T)

    # Compute translation
    t = centroid_closest.T - np.dot(R, centroid_source.T)

    return R, t


def get_boundary_distance(coordinates, arena, return_closest_point=False):
    """
    Compute the distance from each coordinate to the nearest point on the arena boundary.

    Parameters
    ----------
    coordinates : np.ndarray
        Array of shape (..., 2) containing 2D keypoint coordinates.

    arena : np.ndarray
        Array of shape (N, 2) representing the arena as a polygon.

    return_closest_point : bool, default=False
        If True, also return the closest point on the boundary for each coordinate.

    Returns
    -------
    distances : np.ndarray
        Array of shape (...) containing boundary distances.
    """
    arena_polygon = Polygon(arena)
    distances = np.array(
        [Point(coord).distance(arena_polygon.boundary) for coord in coordinates]
    )
    return distances


def point_to_segment_distances(points, segment_start, segment_end):
    """
    Vectorized calculation of distances from multiple points to a single line segment.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (M, 2) containing M points.

    segment_start, segment_end : np.ndarray
        Arrays of shape (2,) representing the start and end points of the segment.

    Returns
    -------
    distances : np.ndarray
        Array of shape (M,) containing distances from each point to the segment.

    closest_points : np.ndarray
        Array of shape (M, 2) containing the closest points on the segment for each input point.
    """
    # Vector from segment start to segment end
    segment_vector = segment_end - segment_start
    # Vector from segment start to points
    point_vectors = points - segment_start

    # Project point vectors onto segment vector
    segment_length_squared = np.sum(segment_vector**2)
    projected_lengths = np.dot(point_vectors, segment_vector) / segment_length_squared
    projected_lengths = np.clip(projected_lengths, 0, 1)

    # Find the closest points on the segment
    closest_points = segment_start + np.outer(projected_lengths, segment_vector)

    # Calculate distances from points to their closest points on the segment
    distances = np.sqrt(np.sum((points - closest_points) ** 2, axis=1))

    return distances, closest_points


def map_to_boundary(coordinates, arena):
    """
    Map each input coordinate point to the nearest point on the arena boundary.

    Parameters
    ----------
    coordinates : np.ndarray
        Array of shape (..., 2) containing 2D keypoint coordinates.

    arena : np.ndarray
        Array of shape (N, 2) representing the arena as a polygon.

    Returns
    -------
    closest_points : np.ndarray
        Array of shape (..., 2) containing the closest points on the boundary for each
        input coordinate point.

    distances : np.ndarray
        Array of shape (...) containing distances to the boundary.
    """
    assert coordinates.shape[-1] == 2, "Coordinates must be 2D"
    shape = coordinates.shape[:-1]
    coordinates = coordinates.reshape(-1, 2)

    num_points = coordinates.shape[0]
    distances = np.full(num_points, np.inf)
    closest_points = np.zeros((num_points, 2))

    for j in range(len(arena)):
        segment_start = arena[j]
        segment_end = arena[(j + 1) % len(arena)]
        segment_distances, segment_closest_points = point_to_segment_distances(
            coordinates, segment_start, segment_end
        )
        update = segment_distances < distances
        distances[update] = segment_distances[update]
        closest_points[update] = segment_closest_points[update]

    return closest_points.reshape(*shape, 2), distances.reshape(*shape)


def egocentric_boundary_direction(coordinates, arena, anterior_idxs, posterior_idxs):
    """
    Compute direction to the nearest boundary point in ego-centric coordinates.

    Parameters
    ----------
    coordinates : np.ndarray
        Array of shape (num_timepoints, num_keypoints, 2) containing keypoints.

    arena : np.ndarray
        Array of shape (N, 2) representing the arena as a polygon.

    anterior_idxs : array-like of int
        Indices of anterior keypoints.

    posterior_idxs : array-like of int
        Indices of posterior keypoints.

    Returns
    -------
    ego_boundary_directions : np.ndarray
        Array of shape (..., 2) containing the egocentric direction of the boundary for
        each input coordinate point.
    """
    assert coordinates.shape[-1] == 2, "Coordinates must be 2D"

    # get centroids and heading vectors
    centroid = coordinates.mean(1)
    anterior_pts = coordinates[:, anterior_idxs].mean(1)
    posterior_pts = coordinates[:, posterior_idxs].mean(1)
    heading = anterior_pts - posterior_pts
    heading /= np.linalg.norm(heading, axis=-1, keepdims=True)

    # get boundary vectors
    closest_point, distance = map_to_boundary(centroid, arena)
    allo_boundary_vector = (centroid - closest_point) / distance[:, None]

    # get angle in egocentric coordinates
    heading_perp = heading[:, ::-1] * np.array([1, -1])
    ego_boundary_directions = (allo_boundary_vector * heading_perp).sum(1)
    return ego_boundary_directions
