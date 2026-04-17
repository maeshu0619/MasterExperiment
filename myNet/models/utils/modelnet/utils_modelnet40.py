import open3d as o3d
import numpy as np
# from util import get_points, set_points, normalize, shuffle_data
import copy


def load_mesh(filepath):
    return o3d.io.read_triangle_mesh(filepath)


def export_mesh(mesh, filepath):
    o3d.io.write_triangle_mesh(filepath, mesh)


def load_pcd(filepath):
    return o3d.io.read_point_cloud(filepath)


def export_pcd(pcd, filepath):
    o3d.io.write_point_cloud(filepath, pcd)


def mesh_to_pcd(mesh, number_of_points=2048):
    return mesh.sample_points_uniformly(number_of_points=number_of_points)



def random_pose(severity):
    """generate a random camera pose"""

    theta = 2 * np.pi * severity / 5
    delta = np.pi / 5
    angle_x = np.random.uniform(2./3. * np.pi, 5./6. * np.pi)
    angle_y = 0
    angle_z = np.random.uniform(theta-delta,theta+delta)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    # a rotation matrix with arbitrarily chosen yaw, pitch, roll
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(-R[:, 2] * 3., 1)  # select the third column, reshape into (3, 1)-vector

    matrix = np.concatenate([np.concatenate([R.T, -np.dot(R.T,t)], 1), [[0, 0, 0, 1]]], 0)
    return matrix

def lidar_pose(severity):
    """generate a random LiDAR pose"""
    theta = 2 * np.pi * severity / 5
    delta = np.pi / 5
    angle_x = 5./8. * np.pi
    angle_y = 0
    angle_z = np.random.uniform(theta-delta,theta+delta)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    # a rotation matrix with arbitrarily chosen yaw, pitch, roll
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(-R[:, 2] * 5, 1)  # select the third column, reshape into (3, 1)-vector
    pose = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
    matrix = np.concatenate([np.concatenate([R.T, -np.dot(R.T,t)], 1), [[0, 0, 0, 1]]], 0)
    return matrix, pose



def get_default_camera_extrinsic():
    return np.array([[1,0,0,1],
                    [0,1,0,0],
                    [0,0,1,2],
                    [0,0,0,1]])


def get_default_camera_intrinsic(width=1920, height=1080):
    return {
        "width": width,
        "height": height,
        "fx": 365,
        "fy": 365,
        "cx": width / 2 - 0.5,
        "cy": height / 2 - 0.5
    }


def core_occlusion(mesh, type, camera_extrinsic=None, camera_intrinsic=None, window_width=1080, window_height=720, n_points=None, downsample_ratio=None):
    if camera_extrinsic is None:
        camera_extrinsic = get_default_camera_extrinsic()
    
    if camera_intrinsic is None:
        camera_intrinsic = get_default_camera_intrinsic()

    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.extrinsic = camera_extrinsic
    camera_parameters.intrinsic.set_intrinsics(**camera_intrinsic)

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(width=window_width, height=window_height)
    viewer.add_geometry(mesh)

    control = viewer.get_view_control()
    control.convert_from_pinhole_camera_parameters(camera_parameters)
    # viewer.run()

    depth = viewer.capture_depth_float_buffer(do_render=True)

    viewer.destroy_window()
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, camera_parameters.intrinsic, extrinsic=camera_parameters.extrinsic)

    if downsample_ratio is not None:
        ratio =  int((1 - downsample_ratio) / downsample_ratio)
        pcd = pcd.uniform_down_sample(ratio)
    elif n_points is not None:
        # print(np.asarray(pcd.points).shape[0])
        ratio =  int(np.asarray(pcd.points).shape[0] / n_points)
        if ratio > 0:
            # if type == 'occlusion':
            set_points(pcd, shuffle_data(np.asarray(pcd.points)))
            pcd = pcd.uniform_down_sample(ratio)
    
    return pcd


def occlusion_1(mesh, type, severity, window_width=1080, window_height=720, n_points=None, downsample_ratio=None):
    points = get_points(mesh)
    points = normalize(points)
    set_points(mesh, points)
    if type == 'occlusion':
        camera_extrinsic = random_pose(severity)
    elif type == 'lidar':
        camera_extrinsic,pose = lidar_pose(severity)
    camera_intrinsic = get_default_camera_intrinsic(window_width, window_height)
    pcd = core_occlusion(mesh, type, camera_extrinsic=camera_extrinsic, camera_intrinsic=camera_intrinsic, window_width=window_width, window_height=window_height, n_points=n_points, downsample_ratio=downsample_ratio)

    points = get_points(pcd)
    if points.shape[0] < n_points:
        index = np.random.choice(points.shape[0], n_points)
        points = points[index]
    # points = normalize(points)
    # points = denomalize(points, scale, offset)
    if type == 'lidar':
        return points[:n_points,:], pose
    else:
        return points[:n_points,:]




def get_points(data):
    if isinstance(data, o3d.cpu.pybind.geometry.TriangleMesh):
        return np.asarray(data.vertices)
    elif isinstance(data, o3d.cpu.pybind.geometry.PointCloud):
        return np.asarray(data.points)
    else:
        raise Exception("Wrong input data format: should be pointcloud or mesh")


def set_points(data, points):
    if isinstance(data, o3d.cpu.pybind.geometry.TriangleMesh):
        data.vertices = o3d.utility.Vector3dVector(points)
        return data
    elif isinstance(data, o3d.cpu.pybind.geometry.PointCloud):
        data.points = o3d.utility.Vector3dVector(points)
        return data
    else:
        raise Exception("Wrong input data format: should be pointcloud or mesh")


def normalize(new_pc):
    new_pc[:,0] -= (np.max(new_pc[:,0]) + np.min(new_pc[:,0])) / 2
    new_pc[:,1] -= (np.max(new_pc[:,1]) + np.min(new_pc[:,1])) / 2
    new_pc[:,2] -= (np.max(new_pc[:,2]) + np.min(new_pc[:,2])) / 2
    leng_x, leng_y, leng_z = np.max(new_pc[:,0]) - np.min(new_pc[:,0]), np.max(new_pc[:,1]) - np.min(new_pc[:,1]), np.max(new_pc[:,2]) - np.min(new_pc[:,2])
    if leng_x >= leng_y and leng_x >= leng_z:
        ratio = 2.0 / leng_x
    elif leng_y >= leng_x and leng_y >= leng_z:
        ratio = 2.0 / leng_y
    else:
        ratio = 2.0 / leng_z
    new_pc *= ratio
    return new_pc


def denomalize(points, scale, offset, hard_copy=False):
    if hard_copy:
        new_points = copy.deepcopy(points)
    else:
        new_points = points

    n_points = new_points.shape[0]
    new_points = new_points * np.tile(scale, (n_points,1)) + np.tile(offset, (n_points,1))
    return new_points

def shuffle_data(data):

    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    return data[idx, ...]


def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def appendCart_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    ptsnew[:,3] = ptsnew[:,0] * np.sin(ptsnew[:,1]) * np.cos(ptsnew[:,2])
    ptsnew[:,4] = ptsnew[:,0] * np.sin(ptsnew[:,1]) * np.sin(ptsnew[:,2])
    ptsnew[:,5] = ptsnew[:,0] * np.cos(ptsnew[:,1]) 
    return ptsnew

    