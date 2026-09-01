"""Utility functions for Polyformer: curvature fitting, FPS, normal estimation.

Not part of the main benchmark pipeline — used for experimental
curvature-based point cloud analysis.
"""
from collections import deque
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import cv2
import numpy as np
import trimesh
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def align_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Align point cloud to the z-axis.
    """

    points = points - points.mean(axis = 0)
    points = points / np.max(np.linalg.norm(points, axis = -1))

    cov = (points.T @ points) / len(points)
    evecs = np.linalg.eigh(cov)[1]

    frame = evecs[:, [2, 1, 0]]

    if np.linalg.det(frame) < 0:
        frame[:, 1] = - frame[:, 1]

    return points @ frame

def farthest_point_sampling(points : np.ndarray, K : int, start : Optional[int] = None, verbose : bool = False) -> np.ndarray:
    """
    points: (N, 3) numpy array
    K: number of points to sample
    return: (K,) numpy array of selected indices
    """
    N = points.shape[0]
    selected = np.zeros(K, dtype=np.int32)
    distances = np.full((N,), float("inf"))

    selected[0] = np.random.randint(0, N) if start is None else start
    centroid = points[selected[0]]

    for i in tqdm(range(1, K), desc = "Farthest point sampling", disable = not verbose, dynamic_ncols = True, smoothing = 0.00):
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        selected[i] = np.argmax(distances)
        centroid = points[selected[i]]

    return selected

def fibonacci_sphere(num_samples: int = 1000, i: Optional[Union[int, Iterable[int]]] = None) -> np.ndarray:

    phi = np.pi * (3 - np.sqrt(5))

    iterable = range(num_samples) if i is None else ([i] if isinstance(i, int) else i)
    points = np.zeros((len(iterable), 3))

    for j, i in enumerate(iterable):
        y = 1 - (i / float(num_samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        points[j, 1] = y

        theta = phi * i  # golden angle increment

        points[j, 0] = np.cos(theta) * radius
        points[j, 2] = np.sin(theta) * radius

    return points

def surface_face_normals(
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        eps: float = 1e-12,
    ):
    """
    Build triangle face normals from a structured surface grid.
    Returns:
        normals:   (2 * (nr - 1) * (nt - 1), 3)
        centroids: (2 * (nr - 1) * (nt - 1), 3)
    """
    V = np.stack([xs, ys, zs], axis=-1)  # (nr, nt, 3)
    # Cell corners
    v00 = V[:-1, :-1, :]
    v10 = V[1:,  :-1, :]
    v01 = V[:-1, 1:,  :]
    v11 = V[1:,  1:,  :]
    # Two triangles per quad: (v00, v10, v11) and (v00, v11, v01)
    n1 = np.cross(v10 - v00, v11 - v00)
    n2 = np.cross(v11 - v00, v01 - v00)
    c1 = (v00 + v10 + v11) / 3.0
    c2 = (v00 + v11 + v01) / 3.0

    normals = np.concatenate([n1.reshape(-1, 3), n2.reshape(-1, 3)], axis=0)
    centroids = np.concatenate([c1.reshape(-1, 3), c2.reshape(-1, 3)], axis=0)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > eps
    normals[valid] /= lengths[valid]
    normals[~valid] = 0.0  # handle degenerate triangles safely

    return normals, centroids

def approximate_curvature(xs, ys, zs):
    # https://hal.science/hal-02529668/document, Section III B, equation (1)
    # https://en.wikipedia.org/wiki/Principal_curvature

    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    zs = zs.reshape(-1)

    (a, b, c, *_), *_ = np.linalg.lstsq(
        a       = np.column_stack((xs**2, xs*ys, ys**2, xs, ys)),
        b       = zs,
        rcond   = None
    )

    hessian = np.array([
        [2*a, b],
        [b, 2*c]
    ])

    (k1, k2), curvature_directions = np.linalg.eigh(hessian)

    return (k1, k2), curvature_directions

def fit_surface(points: np.ndarray, degree = 2, symmetric_only = False):

    terms = []

    xs, ys, zs = np.array_split(points[..., :3], 3, axis = -1)

    for i in range(degree + 1):
        for j in range(degree + 1):
            if i != 0 or j != 0:
                if symmetric_only and i != j:
                    continue
                terms.append(xs**i * ys**j)

    terms = np.column_stack(terms)
    try:
        coefficients, residuals, *_ = np.linalg.lstsq(
            a       = terms,
            b       = zs,
            rcond   = None
        )
    except:
        coefficients = np.zeros(5)
        residuals = np.inf
    finally:
        return coefficients.reshape(-1), residuals

def _surf_zs(xs, ys, coefficients, degree = 2):
    zs = np.zeros_like(xs)

    for i in range(degree + 1):
        for j in range(degree + 1):
            if i != 0 or j != 0:
                zs += coefficients[i * (degree + 1) + j - 1] * xs**i * ys**j

    return zs

def make_surface(
        coefficients: np.ndarray,
        degree = 2,
        r = 0.1,
        num_r = 20,
        num_theta = 20,
        theta_min = 0,
        theta_max = 2 * np.pi
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    rs = np.linspace(0, r, num_r - 1).reshape(-1, 1)
    thetas = np.linspace(theta_min, theta_max, num_theta).reshape(1, -1)

    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)

    zs = _surf_zs(xs, ys, coefficients, degree)

    return xs, ys, zs

def curvature_features(
        points: np.ndarray,
        neighborhoods: Iterable[np.ndarray],
        percentile = 90,
        degree: int = 2,
        show = False,
        num_rotations = 5000,
        min_samples = 30,
        ratio_threshold = 0.2,
        symmetric_only = False,
        *args, **kwargs
    ) -> np.ndarray:

    num_points = len(points)

    def fit_error(points, degree, r):

        coefficients, residual = fit_surface(
            points = points,
            degree = degree
        )

        if np.asarray(residual).size != 1:
            return np.inf, coefficients, (None, None, None)

        xs, ys, zs = make_surface(
            coefficients  = coefficients,
            degree = degree,
            r = r,
            num_r = 10,
            num_theta = 10,
        )

        # zs = _surf_zs(*points[:, :2].T, coefficients, degree)
        # residual = np.abs(zs - points[:, 2]).mean().item()
        # return residual, coefficients, (xs, ys, zs)

        surf_points = np.concatenate([xs.reshape(-1,1), ys.reshape(-1,1), zs.reshape(-1,1)], axis = -1)

        dists = surf_points[:, np.newaxis, :] - points[np.newaxis, 1:, :]
        dists = np.linalg.norm(dists, axis = -1)

        residual = 0

        # 1 -> minimize input point discrepency
        residual += dists.min(axis = 1).mean()

        # 0 -> minimize surface point discrepency
        # residual += dists.min(axis = 0).mean()
        # residual /= 2

        return residual.item(), coefficients, (xs, ys, zs)

    # if symmetric_only:
    #     num_coefficients = degree * 2 + 1
    # else:
    #     num_coefficients = (degree + 1) ** 2 - 1
    
    num_coefficients = (degree + 1) ** 2 - 1

    all_coefficients        = np.zeros((num_points, num_coefficients))
    all_curvature_values    = np.zeros((num_points, 2))
    all_quaternions         = np.zeros((num_points, 4))
    all_normals             = np.zeros((num_points, 3))
    all_rs                  = np.zeros((num_points, 1))

    successes = np.full((num_points), True, dtype = bool)

    rotations = Rotation.random(num_rotations).as_matrix()
    _angles = np.arccos(rotations[:, 2, 2])
    rotations = rotations[np.argsort(_angles)]

    if show:
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(111, projection = '3d')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.ion()
        plt.show(block = False)

    pbar0 = tqdm(total = num_points, desc = "Points", position = 0)
    pbar1 = tqdm(total = len(rotations), desc = "Rotations", position = 1)

    for i in range(num_points):

        # Reset progress bars
        pbar0.update(1)
        pbar1.reset()

        # Initialize best fit parameters
        min_ratio           = np.inf
        min_residual        = np.inf
        min_coefficients    = None
        min_points          = (None, None, None)
        min_transformed     = None
        min_r               = None

        # Get neighborhood and rotations
        neighborhood = neighborhoods[i] - points[i]
        _rotations = rotations

        # Align neighborhood to the z-axis
        _neighborhood = neighborhood - neighborhood.mean(axis = 0)
        evecs = np.linalg.eigh(np.cov(_neighborhood, rowvar = False))[1]
        align = evecs[np.newaxis, :, :]
        if np.linalg.det(align) < 0:
            align[:, 1] = - align[:, 1]
        _rotations = _rotations @ align

        for j, _rotation in enumerate(_rotations):
            pbar1.update(1)

            transformed = neighborhood @ _rotation

            rs = np.linalg.norm(transformed[:, :2], axis = -1)
            r = np.percentile(rs, percentile)

            residual, coefficients, (xs, ys, zs) = fit_error(transformed, degree, r)

            ratio = residual / r

            if ratio < min_ratio:
                min_ratio           = ratio
                min_residual        = residual
                min_coefficients    = coefficients
                min_points          = (xs, ys, zs)
                min_transformed     = transformed
                min_rotation        = _rotation
                min_r               = r

            if j >= min_samples and min_ratio <= ratio_threshold:
                break
        
        if min_coefficients is None:
            successes[i] = False
            continue

        R = Rotation.from_matrix(min_rotation)

        (k1, k2), (v1, v2) = approximate_curvature(*min_points)

        normals, _ = surface_face_normals(
            *make_surface(
                min_coefficients,
                degree = degree,
                r = min_r / 2,
                num_r = 3,
                num_theta = 20
            )
        )
        normal = np.mean(normals, axis = 0)
        normal = (normal / np.linalg.norm(normal)).reshape(-1)

        all_coefficients[i]     = min_coefficients
        all_curvature_values[i] = (k1, k2)
        all_quaternions[i]      = R.as_quat()
        all_normals[i]          = R.apply(normal)
        all_rs[i]               = min_r

        if show:
            ax.cla()

            ax.scatter(*min_transformed.T, color = 'black', alpha = 1.0, s = 1.0)
            ax.plot_surface(*min_points, color = 'red', alpha = 0.5)
            ax.quiver(*np.zeros(3), *normal, color = 'blue', alpha = 1.0, length = 0.05, normalize = True)

            lim = max(r, min_transformed[:, 2].max())
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)

            ax.set_title(f"Residual: {min_residual:.4f}, Ratio: {min_ratio:.2f}")

            plt.draw()
            plt.pause(0.01)

    all_normals, flipped = orient_normals_consistently(points, all_normals, k = 5, seed_idx = 0)

    features = np.concatenate([all_coefficients, all_curvature_values, all_quaternions, all_normals, all_rs], axis = 1)

    return features, successes


def curvature_features(
        points: np.ndarray,
        neighborhoods: Iterable[np.ndarray],
        percentile = 90,
        degree: int = 2,
        show = False,
        num_rotations = 5000,
        min_samples = 30,
        ratio_threshold = 0.2,
        restrict_theta = True,
        symmetric_only = False,
        *args, **kwargs
    ) -> np.ndarray:

    num_points = len(points)

    def fit_error(points, degree, r, restrict_theta = True):

        coefficients, residual = fit_surface(
            points = points,
            degree = degree
        )

        if np.asarray(residual).size != 1:
            return np.inf, coefficients, (None, None, None), (None, None)

        xs, ys, zs = make_surface(
            coefficients  = coefficients,
            degree = degree,
            r = r,
            num_r = 10,
            num_theta = 10,
        )

        # zs = _surf_zs(*points[:, :2].T, coefficients, degree)
        # residual = np.abs(zs - points[:, 2]).mean().item()
        # return residual, coefficients, (xs, ys, zs)

        surf_points = np.concatenate([xs.reshape(-1,1), ys.reshape(-1,1), zs.reshape(-1,1)], axis = -1)

        dists = surf_points[:, np.newaxis, :] - points[np.newaxis, 1:, :]
        dists = np.linalg.norm(dists, axis = -1)

        residual = 0

        # 1 -> minimize input point discrepency
        residual += dists.min(axis = 1).mean()

        # 0 -> minimize surface point discrepency
        residual += dists.min(axis = 0).mean()
        residual /= 2

        theta_min = 0
        theta_max = 2 * np.pi

        if restrict_theta:

            gap_thresh = np.pi / 2

            sorted_thetas = np.sort(np.arctan2(transformed[:, 1], transformed[:, 0]) % (2 * np.pi))
            wraparound_gap = 2 * np.pi - sorted_thetas[-1] + sorted_thetas[0]
            gaps = np.asarray([*np.diff(sorted_thetas), wraparound_gap])

            max_gap_idx = np.argmax(gaps)
            max_gap = gaps[max_gap_idx]

            if max_gap > gap_thresh:

                theta0 = sorted_thetas[max_gap_idx]
                theta1 = sorted_thetas[(max_gap_idx + 1) % len(sorted_thetas)]

                theta_min = theta1
                theta_max = theta0 + 2 * np.pi if theta0 < theta1 else theta0

                # _xs, _ys, _zs = make_surface(
                #     coefficients    = coefficients * 0,
                #     degree          = degree,
                #     r               = r,
                #     num_r           = 10,
                #     num_theta       = 10,
                #     theta_min       = theta_min,
                #     theta_max       = theta_max,
                # )

                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection = '3d')
                
                # # Set view angle to be top down
                # ax.view_init(elev = 90, azim = 0)

                # ax.plot_surface(_xs, _ys, _zs)
                # ax.scatter(*transformed.T, color = 'red', alpha = 1.0, s = 1.0)
                # ax.set_xlabel('X')
                # ax.set_ylabel('Y')
                # ax.set_zlabel('Z')
                # ax.set_aspect('equal')
                # plt.show(block = False)
                # plt.pause(0.1)
                # plt.close()

        return residual.item(), coefficients, (xs, ys, zs), (theta_min, theta_max)

    # if symmetric_only:
    #     num_coefficients = degree * 2 + 1
    # else:
    #     num_coefficients = (degree + 1) ** 2 - 1

    num_coefficients = (degree + 1) ** 2 - 1

    all_coefficients        = np.zeros((num_points, num_coefficients))
    all_curvature_values    = np.zeros((num_points, 2))
    all_quaternions         = np.zeros((num_points, 4))
    all_normals             = np.zeros((num_points, 3))
    all_thetas              = np.zeros((num_points, 2))
    all_rs                  = np.zeros((num_points, 1))

    successes = np.full((num_points), True, dtype = bool)

    rotations = Rotation.random(num_rotations)
    rotations = rotations.as_matrix()[np.argsort(rotations.magnitude())]
    rotations = np.concatenate([np.eye(3).reshape(1, 3, 3), rotations], axis = 0)

    if show:
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(111, projection = '3d')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.ion()
        plt.show(block = False)

    pbar0 = tqdm(total = num_points, desc = "Points", position = 0)
    pbar1 = tqdm(total = len(rotations), desc = "Rotations", position = 1)

    for i in range(num_points):

        # Reset progress bars
        pbar0.update(1)
        pbar1.reset()

        # Initialize best fit parameters
        min_ratio           = np.inf
        min_residual        = np.inf
        min_coefficients    = None
        min_points          = (None, None, None)
        min_transformed     = None
        min_r               = None

        # Get neighborhood and rotations
        neighborhood = neighborhoods[i] - points[i]
        _rotations = rotations

        # Align neighborhood to the z-axis
        _neighborhood = neighborhood# - neighborhood.mean(axis = 0)
        evecs = np.linalg.eigh(np.cov(_neighborhood, rowvar = False))[1]
        align = evecs[:, [2, 1, 0]]
        if np.linalg.det(align) < 0:
            align[:, 1] = - align[:, 1]
        _rotations = _rotations @ align.T[np.newaxis, :, :]

        for j, _rotation in enumerate(_rotations):
            pbar1.update(1)

            transformed = neighborhood @ _rotation

            rs = np.linalg.norm(transformed[:, :2], axis = -1)
            r = np.percentile(rs, percentile)

            residual, coefficients, (xs, ys, zs), (theta_min, theta_max) = fit_error(
                points  = transformed,
                degree  = degree,
                r       = r,
                restrict_theta = restrict_theta
            )

            ratio = residual / r

            if ratio < min_ratio:
                min_ratio           = ratio
                min_residual        = residual
                min_coefficients    = coefficients
                min_points          = (xs, ys, zs)
                min_transformed     = transformed
                min_rotation        = _rotation
                min_r               = r

            if j >= min_samples and min_ratio <= ratio_threshold:
                break
        
        if min_coefficients is None:
            successes[i] = False
            continue

        R = Rotation.from_matrix(min_rotation)

        (k1, k2), (v1, v2) = approximate_curvature(*min_points)

        normals, _ = surface_face_normals(
            *make_surface(
                min_coefficients,
                degree = degree,
                r = min_r / 2,
                num_r = 3,
                num_theta = 20
            )
        )
        normal = np.mean(normals, axis = 0)
        normal = (normal / np.linalg.norm(normal)).reshape(-1)

        all_coefficients[i]     = min_coefficients
        all_curvature_values[i] = (k1, k2)
        all_quaternions[i]      = R.as_quat()
        all_normals[i]          = R.apply(normal)
        all_thetas[i]           = (theta_min, theta_max)
        all_rs[i]               = min_r

        if show:
            ax.cla()

            ax.scatter(*min_transformed.T, color = 'black', alpha = 1.0, s = 1.0)
            ax.plot_surface(*min_points, color = 'red', alpha = 0.5)
            ax.quiver(*np.zeros(3), *normal, color = 'blue', alpha = 1.0, length = 0.05, normalize = True)

            lim = max(r, min_transformed[:, 2].max())
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)

            ax.set_title(f"Residual: {min_residual:.4f}, Ratio: {min_ratio:.2f}")

            plt.draw()
            plt.pause(0.01)

    all_normals, flipped = orient_normals_consistently(points, all_normals, k = 5, seed_idx = 0)
    # TODO orient z axes of transformed coordinate frames consistently

    features = np.concatenate([all_coefficients, all_curvature_values, all_quaternions, all_normals, all_rs, all_thetas], axis = 1)

    return features, successes

def orient_normals_consistently(points: np.ndarray, normals: np.ndarray, k: int = 20, seed_idx: int = 0) -> np.ndarray:
    """
    Orient point cloud normal vectors so they are locally consistent.
    """

    neighbors = KDTree(points).query(points, k = k + 1)[1][:, 1:]

    normals = normals.copy()
    num_points = points.shape[0]
    flipped = np.full(num_points, -1)

    visited = np.zeros(num_points, dtype = bool)
    visited[seed_idx] = True

    queue = deque([seed_idx])

    while queue:
        i = queue.popleft()
        ni = normals[i]

        for j in neighbors[i]:
            if visited[j]:
                continue

            nj = normals[j]

            # Flip normal if inconsistent
            if np.dot(ni, nj) < 0:
                normals[j] *= -1
                flipped[j] *= -1

            visited[j] = True
            queue.append(j)

    # Correct so that the majority of the normal vectors point away from the origin
    num_neg = np.sum(np.sum(np.multiply(normals, points), axis = 1) < 0)

    if num_neg > num_points / 2:
        normals *= -1
        flipped *= -1

    return normals, flipped

def plot_neighborhoods(neighborhoods: Iterable[np.ndarray], ax = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
    else:
        fig = ax.get_figure()

    for neighborhood in neighborhoods:
        ax.scatter(*neighborhood.T, color = np.random.rand(3), alpha = 0.5)

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Neighborhoods')

    return fig, ax

def plot_frames(points: np.ndarray, frames: np.ndarray, ax = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
    else:
        fig = ax.get_figure()

    for point, frame in zip(points, frames):
        ax.quiver(*point.T, *frame[0].T, color = 'red', alpha = 1.0, length = 0.05, normalize = True)
        ax.quiver(*point.T, *frame[1].T, color = 'green', alpha = 1.0, length = 0.05, normalize = True)
        ax.quiver(*point.T, *frame[2].T, color = 'blue', alpha = 1.0, length = 0.05, normalize = True)

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Frames')

    return fig, ax

if __name__ == "__main__":

    np.random.seed(42)

    num_points = 128
    degree = 4
    percentile = 80
    num_neighbors = 32

    # Curvature feature Calculation Parameters

    num_rotations   = 500
    min_samples     = 25
    ratio_threshold = 0.2
    symmetric_only  = False
    restrict_theta  = False

    model = (
        "knight"
        # "chair"
        # "eros"
        # "50-satellite"
        # "75-satellite"
        # "car"
        # "sphere"
        # "dsco"
        # "cloudsat"
    )

    model_dir = Path("data", "models")

    sphere = False
    if model == "knight":
        mesh = trimesh.load(model_dir / "knight" / "12930_WoodenChessKnightSideA_v1_l3.obj")
        points = mesh.vertices
        points = points[farthest_point_sampling(points, 8192)]
    elif model == "eros":
        mesh = trimesh.load(model_dir / "eros" / "eros-16384-sz5.ply", force = "mesh")
        points = mesh.vertices
    elif model == "sphere":
        sphere = True
        downsample_factor = 32
        points = fibonacci_sphere(num_points * downsample_factor)
    else:
        if model == "50-satellite":
            fp = ["50-satellite", "satellite_obj.obj"]
        elif model == "75-satellite":
            fp = ["75-satellite", "Satellite", "Satellite.obj"]
        elif model == "car":
            fp = ["car", "car.glb"]
        elif model == "dsco":
            fp = ["Deep Space Climate Observatory (DSCOVR) (Triana)", "Deep Space Climate Observatory (DSCOVR) (Triana).glb"]
        elif model == "cloudsat":
            fp = ["CloudSat (B)", "cloudsat.ply"]
        else:
            raise NotImplementedError(f"Model {model} not implemented")

        mesh = trimesh.load(model_dir / Path(*fp), force = "mesh")
        points = mesh.vertices
        points = points - np.asarray(mesh.bounds).mean(axis = 0)
        points = points / np.max(np.linalg.norm(points, axis = -1))

        points, *_ = trimesh.remesh.subdivide_to_size(
            vertices    = points,
            faces       = mesh.faces,
            max_edge    = 0.05,
        )

        points = points[farthest_point_sampling(points, 8192)]

    points = align_point_cloud(points)

    all_points = points
    points = points[farthest_point_sampling(points, num_points)]

    tree = KDTree(all_points)

    _, neighborhoods = tree.query(points, k = num_neighbors)
    neighborhoods = [all_points[neighborhood] for neighborhood in neighborhoods]

    # plot_neighborhoods(neighborhoods)
    # plt.show()
    # plt.close()

    features, successes = curvature_features(
        points          = points.copy(),
        neighborhoods   = neighborhoods,
        percentile      = percentile,
        num_rotations   = num_rotations,
        min_samples     = min_samples,
        ratio_threshold = ratio_threshold,
        symmetric_only  = symmetric_only,
        degree          = degree,
        restrict_theta  = restrict_theta,
        show            = False,
    )

    coefficients, curvature_values, quaternions, normals, radii, thetas = np.array_split(features, [-12, -10, -6, -3, -2], axis = 1)

    points = points[successes]

    mean_curvature      = np.mean(curvature_values, axis = 1)
    curvature_energy    = np.abs(np.multiply(curvature_values[:, 0], curvature_values[:, 1]))

    color_metric = np.log(curvature_energy)

    cv2.normalize(color_metric, color_metric, 0, 255, cv2.NORM_MINMAX)
    colors = cv2.applyColorMap(color_metric.astype(np.uint8), cv2.COLORMAP_JET) / 255.0

    fig = plt.figure()
    show_points     = True
    show_normals    = True

    if show_points:
        if show_normals:
            ax          = fig.add_subplot(131, projection = '3d')
            ax_points   = fig.add_subplot(132, projection = '3d')
            ax_normals  = fig.add_subplot(133, projection = '3d')
            axes        = [ax_points, ax, ax_normals]
            titles      = ["Points", 'Surface', 'Normals']
        else:
            ax          = fig.add_subplot(121, projection = '3d')
            ax_points   = fig.add_subplot(122, projection = '3d')
            axes        = [ax_points, ax]
            titles      = ['Points', 'Surface']

    elif show_normals:
        ax          = fig.add_subplot(121, projection = '3d')
        ax_normals  = fig.add_subplot(122, projection = '3d')
        axes        = [ax, ax_normals]
        titles      = ['Surface', 'Normals']

    else:
        ax          = fig.add_subplot(111, projection = '3d')
        axes        = [ax]
        titles      = ['Surface']

    for point, coef, neighborhood, quaternion, color, normal, r, (theta_min, theta_max) in zip(points, coefficients, neighborhoods, quaternions, colors, normals, radii, thetas):

        if np.linalg.norm(quaternion) < 1e-6:
            continue

        R = Rotation.from_quat(quaternion)

        if show_points:
            ax_points.scatter(*point.T, color = color, alpha = 1.0, s = 1.0)
        
        xs, ys, zs = make_surface(coef, degree = degree, r = r, num_r = 5, num_theta = 12, theta_min = theta_min, theta_max = theta_max)
        shape0 = xs.shape

        surf_points = np.concatenate([xs.reshape(-1,1), ys.reshape(-1,1), zs.reshape(-1,1)], axis = -1)
        surf_points = R.apply(surf_points) + point

        ax.plot_surface(*surf_points.T.reshape(3, *shape0), color = color, alpha = 0.5)

        if show_normals:
            ax_normals.quiver(*point.T, *normal.T, color = 'black', alpha = 1.0, length = 0.1, normalize = True)

    for _ax, title in zip(axes, titles):
        _ax.set_title(title)
        _ax.set_xlim(-1, 1)
        _ax.set_ylim(-1, 1)
        _ax.set_zlim(-1, 1)
        _ax.set_aspect('equal')
        _ax.set_xlabel('X')
        _ax.set_ylabel('Y')
        _ax.set_zlabel('Z')

    # Set view position
    el0 = 30
    az0 = 45
    
    for _ax in axes:
        _ax.view_init(elev = el0, azim = az0)

    i = 0
    inc = 1

    plt.show(block = False)

    while True:
        plt.pause(0.01)
        i += inc

        # if i > 90 or i < 0:
        #     inc = -inc

        for _ax in axes:
            _ax.view_init(elev = el0, azim = az0 + i)

        # Check if the window is closed
        if not plt.fignum_exists(fig.number):
            break

    plt.close()
