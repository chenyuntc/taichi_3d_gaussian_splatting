import numpy as np
import taichi as ti
import taichi.math as tm
import torch
from Camera import CameraInfo

data_type = ti.f32
torch_type = torch.float32


@ti.func
def intersect_ray_with_ellipsoid(
    ray_origin: tm.vec3,
    ray_direction: tm.vec3,
    ellipsoid_R: tm.mat3,  # R
    ellipsoid_t: tm.vec3,
    ellipsoid_S: tm.vec3,
    eps: ti.f32 = 1e-5
):
    """ intersect a ray with an ellipsoid

    Args:
        ray_origin (tm.vec3): the origin of the ray in the world space
        ray_direction (tm.vec3): the direction of the ray in the world space
        ellipsoid_R (tm.mat3): the rotation matrix of the ellipsoid
        ellipsoid_S (tm.vec3): the scale of the ellipsoid
        eps (ti.f32, optional): _description_. Defaults to 1e-5.

    Returns:
        (ti.i32, tm.vec3): whether the ray intersects with the ellipsoid, and the intersection point in the world space
    """
    o = ray_origin
    d = ray_direction
    t = ellipsoid_t
    R = ellipsoid_R
    S = ellipsoid_S
    has_intersection = False
    intersection_point = tm.vec3(0.0, 0.0, 0.0)

    inv_transform_matrix = tm.mat3([
        [1 / S[0], 0, 0],
        [0, 1 / S[1], 0],
        [0, 0, 1 / S[2]]
    ]) @ (R.transpose())
    o_transformed = inv_transform_matrix @ (o - t)
    d_transformed = inv_transform_matrix @ d

    A = tm.dot(d_transformed, d_transformed)
    if abs(A) < eps:
        A = eps

    B = 2 * tm.dot(o_transformed, d_transformed)
    C = tm.dot(o_transformed, o_transformed) - 1

    discriminant = B ** 2 - 4 * A * C
    if discriminant < 0:
        has_intersection = False
    else:
        if abs(discriminant) < eps:
            discriminant = 0

        sqrt_discriminant = ti.sqrt(discriminant)
        t1 = (-B - sqrt_discriminant) / (2 * A)
        t2 = (-B + sqrt_discriminant) / (2 * A)

        if t1 < 0 and t2 < 0:
            has_intersection = False
        else:
            t_intersect = t1 if t1 >= 0 else t2
            if abs(t1 - t2) < eps:
                t_intersect = ti.min(t1, t2)

            intersection_point_transformed = o_transformed + t_intersect * d_transformed
            transform_mat = R @ tm.mat3([
                [S[0], 0, 0],
                [0, S[1], 0],
                [0, 0, S[2]]])
            intersection_point = transform_mat @ intersection_point_transformed + t

            has_intersection = True
    return has_intersection, intersection_point


@ti.func
def get_point_to_line_vector(
    point: tm.vec3,
    line_origin: tm.vec3,
    line_direction: tm.vec3
):
    """ given a point and a line, return the vector from the point to the line

    Args:
        point (tm.vec3): the point, x, y, z
        line_origin (tm.vec3): the origin of the line(ray) in the same space as the point, x, y, z
        line_direction (tm.vec3): the direction of the line(ray), x, y, z

    Returns:
        tm.vec3: the vector from the point to the line
    """
    p = point
    o = line_origin
    d = line_direction
    op = p - o
    scale_factor = ti.math.dot(op, d) / ti.math.dot(d, d)
    q = o + scale_factor * d
    qp = p - q
    return qp


def get_ray_origin_and_direction_from_camera(
    T_pointcloud_camera: torch.Tensor,
    camera_info: CameraInfo
):
    """ get the ray origin and direction from the camera

    Args:
        T_pointcloud_camera (torch.Tensor): 4x4 SE(3) matrix, transforms points from the camera frame to the pointcloud frame
        camera_info (CameraInfo): the camera info

    Returns:
        (torch.Tensor, torch.Tensor): the ray origin and direction, ray_origin: (3,), direction: (H, W, 3)
    """
    T_camera_pointcloud = inverse_se3(T_pointcloud_camera)
    ray_origin = T_pointcloud_camera[:3, 3]
    """ consider how we get a point(x, y, z)'s position in the camera frame:
    p_in_camera_frame = T_camera_pointcloud * [x, y, z, 1]^T
    let p_in_camera_frame be [x', y', z', 1]^T
    |u|   |fx  0  cx|   |x'/z'|                     
    |v| = |0   fy cy| * |y'/z'|, where u, v are the pixel coordinates in the image
    |1|   |0   0  1 |   | 1  |
    now, we want to get the direction of the ray, which is the vector from the camera center to the point
    we take z' = 1, so we have:
    |x'|   |fx  0  cx|^-1   |u|
    |y'| = |0   fy cy|    * |v|
    |1 |   |0   0  1 |      |1|
    , where x', y' are the coordinates of the point in the camera frame, 
    then we can get the direction of the ray by:
    [x, y, z, 1]^T = T_pointcloud_camera * [x', y', 1, 1]^T
    direction = [x, y, z, 1]^T - ray_origin
    """
    pixel_u, pixel_v = torch.meshgrid(torch.arange(
        camera_info.camera_width), torch.arange(camera_info.camera_height))
    pixel_u, pixel_v = pixel_u.float(), pixel_v.float()
    pixel_u += 0.5  # add 0.5 to make the pixel coordinates be the center of the pixel
    pixel_v += 0.5  # add 0.5 to make the pixel coordinates be the center of the pixel
    pixel_uv_1 = torch.stack(
        [pixel_u, pixel_v, torch.ones_like(pixel_u)], dim=-1)  # (H, W, 3)
    pixel_uv_1 = pixel_uv_1.reshape(-1,
                                    3).to(camera_info.camera_intrinsics.device)
    fx = camera_info.camera_intrinsics[0, 0]
    fy = camera_info.camera_intrinsics[1, 1]
    cx = camera_info.camera_intrinsics[0, 2]
    cy = camera_info.camera_intrinsics[1, 2]
    inv_camera_intrinsics = torch.tensor([
        [1 / fx, 0, -cx / fx],
        [0, 1 / fy, -cy / fy],
        [0, 0, 1]], dtype=camera_info.camera_intrinsics.dtype, device=camera_info.camera_intrinsics.device)
    pixel_xy_1 = inv_camera_intrinsics @ pixel_uv_1.T  # (3, H*W)
    pixel_xy_1 = pixel_xy_1.T  # (H*W, 3)
    pixel_xy_1 = torch.cat([pixel_xy_1, torch.ones_like(
        pixel_xy_1[:, :1])], dim=-1)  # (H*W, 4)
    pixel_xyz_1 = T_camera_pointcloud @ pixel_xy_1.T  # (4, H*W)
    pixel_xyz_1 = pixel_xyz_1.T  # (H*W, 4)
    pixel_xyz = pixel_xyz_1[:, :3].reshape(
        camera_info.camera_height, camera_info.camera_width, 3)  # (H, W, 3)
    direction = pixel_xyz - ray_origin.reshape(1, 1, 3)
    return ray_origin, direction


@ti.func
def get_point_probability_density_from_2d_gaussian(
    xy: ti.math.vec2,
    gaussian_mean: ti.math.vec2,
    gaussian_covariance: ti.math.mat2,
) -> ti.f32:
    xy_mean = xy - gaussian_mean
    det_cov = gaussian_covariance.determinant()
    inv_cov = (1. / det_cov) * \
        ti.math.mat2([[gaussian_covariance[1, 1], -gaussian_covariance[0, 1]],
                      [-gaussian_covariance[1, 0], gaussian_covariance[0, 0]]])
    xy_mean_T_cov_inv = xy_mean @ inv_cov
    xy_mean_T_cov_inv_xy_mean = xy_mean_T_cov_inv @ xy_mean
    exponent = -0.5 * xy_mean_T_cov_inv_xy_mean
    return ti.exp(exponent) / (2 * np.pi * ti.sqrt(det_cov))


@ti.func
def grad_point_probability_density_2d(
    xy: ti.math.vec2,
    gaussian_mean: ti.math.vec2,
    gaussian_covariance: ti.math.mat2,
):
    xy_mean = xy - gaussian_mean
    det_cov = gaussian_covariance.determinant()
    inv_cov = (1. / det_cov) * \
        ti.math.mat2([[gaussian_covariance[1, 1], -gaussian_covariance[0, 1]],
                      [-gaussian_covariance[1, 0], gaussian_covariance[0, 0]]])
    cov_inv_xy_mean = inv_cov @ xy_mean
    xy_mean_T_cov_inv_xy_mean = xy_mean @ cov_inv_xy_mean
    exponent = -0.5 * xy_mean_T_cov_inv_xy_mean
    p = ti.exp(exponent) / (2 * np.pi * ti.sqrt(det_cov))
    d_p_d_mean = p * cov_inv_xy_mean
    xy_mean_outer_xy_mean = xy_mean.outer_product(xy_mean)
    d_p_d_cov = -0.5 * p * (inv_cov - inv_cov @
                            xy_mean_outer_xy_mean @ inv_cov)
    return p, d_p_d_mean, d_p_d_cov


@ti.kernel
def torch2ti(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        field[I] = data[I]


@ti.kernel
def ti2torch(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = field[I]


@ti.kernel
def ti2torch_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        grad[I] = field.grad[I]


@ti.kernel
def torch2ti_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        field.grad[I] = grad[I]


def inverse_se3(transform: torch.Tensor):
    R = transform[:3, :3]
    t = transform[:3, 3]
    inverse_transform = torch.zeros_like(transform)
    inverse_transform[:3, :3] = R.T
    inverse_transform[:3, 3] = -R.T @ t
    inverse_transform[3, 3] = 1
    return inverse_transform