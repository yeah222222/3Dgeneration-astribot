import os
import numpy as np

from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

def get_camera_to_world_matrix(RT):
    # Convert 3x4 RT matrix to 4x4 matrix
    RT_4x4 = np.eye(4)
    RT_4x4[:3, :4] = np.array(RT)

    # Compute the inverse of the 4x4 matrix
    RT_4x4_inv = np.linalg.inv(RT_4x4)

    # Convert back to 3x4 matrix
    RT_inv = RT_4x4_inv[:3, :]

    return RT_inv
def expand_to_4x4(matrix_3x4):
    """
    将一个 3x4 矩阵扩展为 4x4 矩阵。
    
    参数:
    matrix_3x4: 一个 3x4 的 numpy 数组
    
    返回值:
    一个 4x4 的 numpy 数组
    """
    matrix_4x4 = np.eye(4)
    matrix_4x4[:3, :4] = matrix_3x4
    return matrix_4x4
# 将depth map反投影至三维空间
def depth_image_to_point_cloud(rgb, depth, scale, K, pose):
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)
    
    # K为内参矩阵3*3
    # 图片坐标转相机坐标
    depth[depth == np.max(depth)] = 0
    Z = depth.astype(float) / scale
   
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    X = np.ravel(X)
    Y = -np.ravel(Y)   # Blender的坐标系为[x, -y, -z]
    Z = -np.ravel(Z)
    valid = (Z < 0)
    # X = -np.ravel(X)
    # Y = np.ravel(Y)   # Blender的坐标系为[x, -y, -z]
    # Z = np.ravel(Z)
    # valid = (Z > 0)
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    position = np.vstack((X, Y, Z, np.ones(len(X))))
    
    # 相机坐标转世界坐标
    transform = np.array([[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],    
                          [0, 0, 0, 1]])
    pose = np.dot(transform, pose)
    print('pose: ',pose)
    position = np.dot(pose, position)
    
    R = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]
    # R = np.ravel(rgb[:, :, 0])
    # G = np.ravel(rgb[:, :, 1])
    # B = np.ravel(rgb[:, :, 2])
    
    print(position.shape, R.shape)
    points = np.transpose(np.vstack((position[:3, :], R, G, B))).tolist()

    return points


# 将点云写入ply文件
def write_point_cloud(ply_filename, points):
    formatted_points = []
    for point in points:
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3], point[4], point[5]))

    out_file = open(ply_filename, "w")
    out_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()
    
    
# image_files: XXXXXX.png (RGB, 24-bit, PNG)
# depth_files: XXXXXX.png (16-bit, PNG)
# poses: camera-to-world, 4×4 matrix in homogeneous coordinates
def build_point_cloud(camera_intrinsics, scale, view_ply_in_world_coordinate, poses, image_files,depth_files):

    K = camera_intrinsics
    # print(K)
    # print(poses)

    sum_points_3D = []
    for i in tqdm(range(0, len(image_files))):
        image_file = image_files[i]
        depth_file = depth_files[i]

        rgb = cv2.imread(image_file)
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)#.astype(np.uint16)
        depth = depth[..., 0]
        depth[depth == np.max(depth)] = 0
        if view_ply_in_world_coordinate:
            current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses[i])
        else:
            current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses[i])
#             print(len(current_points_3D), current_points_3D[0])
        save_ply_path = os.path.join("/home/stardust/zjc/r_depth_pointcloud", "point_clouds_exr_testz_zjc_gg_NEG223311_revise_extrinsics_again")

        if not os.path.exists(save_ply_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.mkdir(save_ply_path)
        save_ply_name = os.path.basename(os.path.splitext(image_files[i])[0]) + ".ply"

        write_point_cloud(os.path.join(save_ply_path, save_ply_name), current_points_3D)
        sum_points_3D.extend(current_points_3D)
    write_point_cloud(os.path.join(save_ply_path, 'all_exr_1.0_scale_test.ply'), sum_points_3D)

gemini1920_camera_matrix = np.array([[1038.97339, 0, 964.007812],
                                     [0, 1039.27478, 547.759949],
                                     [0, 0, 1]])


poses = []
name = 'output_zjc_testz_nolimitcamera_gg_neg223311_revise_extrinsics_again'
no_of_name = '001'
d_pose = os.path.join('/home/stardust/zjc',name,no_of_name)
d = os.path.join('/home/stardust/zjc/render/home/stardust/zjc',name,no_of_name)
depth_d = os.path.join('/home/stardust/zjc/render/home/stardust/zjc',name, no_of_name)
# for f in os.listdir(d_pose):
#     if f.endswith('.npy') and f.startswith('001'):
#         print(f)
#         p = np.load(os.path.join(d_pose,f))
#         # p = get_camera_to_world_matrix(p)
#         p = expand_to_4x4(p)
#         # print(p)
#         poses.append(p)

image_files = []
depth_files = []

# for img in os.listdir(d):
#     if img.endswith('_albedo0001.png'):
#         image_files.append(os.path.join(d,img))
#         depth_files.append(os.path.join(depth_d,img.split('_albedo0001.png')[0]+'_depth0001.exr'))
#         npy_array = np.load(os.path.join(d_pose,img.split('_albedo0001.png')[0])+'.npy')
#         print('npy file path: ',os.path.join(d_pose,img.split('_albedo0001.png')[0])+'.npy')
#         npy_array = expand_to_4x4(npy_array)
#         poses.append(npy_array)
for img in os.listdir(d):
    if img.endswith('_albedo0001.png'):
        image_files.append(os.path.join(d_pose,img.split('_albedo0001.png')[0]+'.png'))
        print('############image file path:################', os.path.join(d_pose,img.split('_albedo0001.png')[0]+'.png'))
        depth_files.append(os.path.join(depth_d,img.split('_albedo0001.png')[0]+'_depth0001.exr'))
        npy_array = np.load(os.path.join(d_pose,img.split('_albedo0001.png')[0])+'.npy')
        print('npy file path: ',os.path.join(d_pose,img.split('_albedo0001.png')[0])+'.npy')
        npy_array = expand_to_4x4(npy_array)
        poses.append(npy_array)
# image_files = image_files[:5]
# # print(image_files)
# depth_files = depth_files[:5]
# print(depth_files)
build_point_cloud(camera_intrinsics=gemini1920_camera_matrix, scale=1.0,view_ply_in_world_coordinate=True, poses=poses, image_files=image_files, depth_files=depth_files)