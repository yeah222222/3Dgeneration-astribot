import math
import os
import sys
import numpy as np
import argparse
import json
import random
import time
import bpy
from mathutils import Vector, Matrix
import csv
import mathutils
INPUT_DIR = '/home/stardust/zjc/normalized_model_grasp'
OUTPUT_DIR = '/home/stardust/zjc/output_zjc_testz_nolimitcamera_gg_neg223311_revise_extrinsics_again'
ENGINE = "CYCLES"
SCALE = 0.8
FORMAT = 'OPEN_EXR'
COLOR_DEPTH = '16'
# DEPTH_SCALE = 0.5#待定

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

def load_xyz(filename):

    with open(filename, 'r') as file:
        lines = file.readlines()

    coordinates = []
    
    for line in lines:
        parts = line.split()
        x, y, z = map(float, parts)
        coordinates.append([x, y, z])
    return np.array(coordinates)

CAMERA_POSE = load_xyz("/home/stardust/zjc/render/camera_pose.xyz")
CAMERA_POSE = CAMERA_POSE[:6]
# CAMERA_POSE = np.array([[0.5,0,2],[0,0.5,2],[0,2,3]])#z:[[0,0,1.4],[0,0,3.4]]       y: [[0,1.4,0],[0,3.4,0]]    x: [[1.4,0,0],[3.4,0,0]] 
print('===================', ENGINE, '===================')



# 给定的内参矩阵和畸变系数
gemini1920_camera_matrix = np.array([[1038.97339, 0, 964.007812],
                                     [0, 1039.27478, 547.759949],
                                     [0, 0, 1]])
gemini1920_dist_coeffs = np.array([0.00784674939, -0.0513733439, 0.0000274930753, -0.000323233951, 0.0343934223])

# 内参解析
fx = gemini1920_camera_matrix[0, 0]
fy = gemini1920_camera_matrix[1, 1]
cx = gemini1920_camera_matrix[0, 2]
cy = gemini1920_camera_matrix[1, 2]


# 设置场景和相机
context = bpy.context
scene = context.scene
render = scene.render



# 设置光照
bpy.ops.object.light_add(type="AREA")
light2 = bpy.data.lights["Area"]
light2.energy = 3000
bpy.data.objects["Area"].location[2] = 0.5
bpy.data.objects["Area"].scale[0] = 100
bpy.data.objects["Area"].scale[1] = 100
bpy.data.objects["Area"].scale[2] = 100

# 渲染设置
render.engine = 'CYCLES'
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 1920
render.resolution_y = 1080
render.resolution_percentage = 100
render.film_transparent = True

scene.cycles.device = "GPU"
scene.cycles.samples = 128 
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True


cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)

# 设置相机焦距和传感器尺寸

cam.data.sensor_width = 32
cam.data.sensor_height = 32 * render.resolution_y / render.resolution_x


cam.data.lens = (fx+fy)/2 * cam.data.sensor_width/render.resolution_x
cam.data.shift_x = (render.resolution_x/2 - cx)/render.resolution_x
cam.data.shift_y = (cy - render.resolution_y/2)/render.resolution_x
# 设置相机约束
cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"
# 设置计算设备类型
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

# 启用节点编辑器和镜头畸变效果
scene.use_nodes = True
tree = scene.node_tree
links = tree.links

# 清除现有节点
for node in tree.nodes:
    tree.nodes.remove(node)

scene.use_nodes = True
scene.view_layers["ViewLayer"].use_pass_normal = True
scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
scene.view_layers["ViewLayer"].use_pass_object_index = True
scene.view_layers["ViewLayer"].use_pass_z = True #https://projects.blender.org/blender/blender/issues/100417


nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links

# Clear default nodes
for n in nodes:
    nodes.remove(n)

# Create input render layer node
render_layers = nodes.new('CompositorNodeRLayers')

# Create depth output nodes
depth_file_output = nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
depth_file_output.base_path = ''
depth_file_output.file_slots[0].use_node_format = True
depth_file_output.format.file_format = FORMAT
depth_file_output.format.color_depth = COLOR_DEPTH
if FORMAT == 'OPEN_EXR':
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
else:
    depth_file_output.format.color_mode = "BW"

    # Remap as other types can not represent the full range of depth.
    map = nodes.new(type="CompositorNodeMapValue")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.offset = [-0.7]
    map.size = [DEPTH_SCALE]
    map.use_min = True
    map.min = [0]

    links.new(render_layers.outputs['Depth'], map.inputs[0])
    links.new(map.outputs[0], depth_file_output.inputs[0])

# Create normal output nodes
scale_node = nodes.new(type="CompositorNodeMixRGB")
scale_node.blend_type = 'MULTIPLY'
# scale_node.use_alpha = True
scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
links.new(render_layers.outputs['Normal'], scale_node.inputs[1])

bias_node = nodes.new(type="CompositorNodeMixRGB")
bias_node.blend_type = 'ADD'
# bias_node.use_alpha = True
bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
links.new(scale_node.outputs[0], bias_node.inputs[1])

normal_file_output = nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = 'Normal Output'
normal_file_output.base_path = ''
normal_file_output.file_slots[0].use_node_format = True
normal_file_output.format.file_format = FORMAT
links.new(bias_node.outputs[0], normal_file_output.inputs[0])

# Create albedo output nodes
alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])

albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = 'Albedo Output'
albedo_file_output.base_path = ''
albedo_file_output.file_slots[0].use_node_format = True
albedo_file_output.format.file_format = 'PNG'
albedo_file_output.format.color_mode = 'RGBA'
albedo_file_output.format.color_depth = COLOR_DEPTH
links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])

# Create id map output nodes
id_file_output = nodes.new(type="CompositorNodeOutputFile")
id_file_output.label = 'ID Output'
id_file_output.base_path = ''
id_file_output.file_slots[0].use_node_format = True
id_file_output.format.file_format = FORMAT
id_file_output.format.color_depth = COLOR_DEPTH

if FORMAT == 'OPEN_EXR':
    links.new(render_layers.outputs['IndexOB'], id_file_output.inputs[0])
else:
    id_file_output.format.color_mode = 'BW'

    divide_node = nodes.new(type='CompositorNodeMath')
    divide_node.operation = 'DIVIDE'
    divide_node.use_clamp = False
    divide_node.inputs[1].default_value = 2**int(COLOR_DEPTH)

    links.new(render_layers.outputs['IndexOB'], divide_node.inputs[0])
    links.new(divide_node.outputs[0], id_file_output.inputs[0])

def getCameraIntrinsics(cam_data):
    """
    获取相机的视场角和纵横比
    """
    fov = cam_data.angle_y  # 垂直视场角，单位为弧度
    scene = bpy.context.scene
    w = scene.render.resolution_x  # 渲染分辨率宽度
    h = scene.render.resolution_y  # 渲染分辨率高度
    aspect = w / h  # 纵横比
    return fov, aspect

def computeIntrinsicsMatrix(fov, aspect, width, height):
    """
    计算相机的内参矩阵
    """
    # 计算焦距（单位：像素）
    fy = 1/math.tan(fov*0.5)
    fx = fy / aspect
    
    # 计算主点坐标（通常是图像中心）
    cx = width / 2.0
    cy = height / 2.0

    # 构建内参矩阵
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    
    return K

def saveMatrixToCSV(matrix, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(matrix)

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def load_object_prev(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")

def load_object(object_path: str, location=(0, 0, 0), rotation=(0, 0, 0)) -> None:
    """Loads a glb model into the scene and adjusts its orientation so that the head is facing up."""
    # Load the model
    if object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")
    

    # return obj
def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def scene_bbox(single_obj=None, ignore_matrix=False): #20240803 ignore_matrix =False
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def normalize_scene(obj_path):
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    # bpy.ops.import_scene.obj(filepath=obj_path)
    # obj = bpy.context.selected_objects[0]
    context.view_layer.objects.active = obj

def randomize_lighting() -> None:
    light2.energy = random.uniform(300, 600)
    bpy.data.objects["Area"].location[0] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[1] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[2] = random.uniform(0.5, 1.5)

def set_camera_location(cam_pose, i):
    radius = 1.4 
    x = cam_pose[0]
    y = cam_pose[1]
    z = cam_pose[2] 

    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z
    # if i == 0:


    # 设置相机的旋转
    direction = -camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler() #[0,0,0] #

    print(camera.rotation_euler)


        # 返回相机对象及生成的坐标
        
    # else:
    #     direction = (0,0,-1.4)
    #     direction = mathutils.Vector(direction)
    #     rot_quat = direction.to_track_quat('-Z', 'Y')
    #     camera.rotation_euler = rot_quat.to_euler()  
    return camera, x, y, z
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

# def get_calibration_matrix_K_from_blender(camd):
#     print(camd.type)
#     if camd.type != 'PERSP':
#         raise ValueError('Non-perspective cameras not supported')
#     scene = bpy.context.scene
#     f_in_mm = camd.lens
#     scale = scene.render.resolution_percentage / 100
#     resolution_x_in_px = scale * scene.render.resolution_x
#     resolution_y_in_px = scale * scene.render.resolution_y
#     sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
#     sensor_fit = get_sensor_fit(
#         camd.sensor_fit,
#         scene.render.pixel_aspect_x * resolution_x_in_px,
#         scene.render.pixel_aspect_y * resolution_y_in_px
#     )
#     pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
#     if sensor_fit == 'HORIZONTAL':
#         view_fac_in_px = resolution_x_in_px
#     else:
#         view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
#     pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
#     s_u = 1 / pixel_size_mm_per_px
#     s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

#     # Parameters of intrinsic calibration matrix K
#     u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
#     v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
#     skew = 0 # only use rectangular pixels

#     K = Matrix(
#         ((s_u, skew, u_0),
#         (   0,  s_v, v_0),
#         (   0,    0,   1)))
#     return K
# def get_3x4_RT_matrix_from_blender(cam):
    # # bcam stands for blender camera
    # R_blender2shapenet = Matrix(
    #     ((1, 0, 0),
    #      (0, 0, -1),
    #      (0, 1, 0)))

    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, -1, 0),
    #     (0, 0, -1)))


    # location, rotation = cam.matrix_world.decompose()[0:2]
    # R_world2bcam = rotation.to_matrix().transposed()


    # T_world2bcam = -1*R_world2bcam * location

    # R_world2cv = R_bcam2cv*R_world2bcam*R_blender2shapenet
    # T_world2cv = R_bcam2cv*T_world2bcam

    # # put into 3x4 matrix
    # RT = Matrix((
    #     R_world2cv[0][:] + (T_world2cv[0],),
    #     R_world2cv[1][:] + (T_world2cv[1],),
    #     R_world2cv[2][:] + (T_world2cv[2],)
    #     ))
    # return RT
########################################################################worldto camera #######################################
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera

    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    print("R_world2bcam:  ", R_world2bcam)

    T_world2bcam = -1*R_world2bcam @ location
    print("T_world2bcam:  ", T_world2bcam)

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT
########################################################################camera to world #######################################
def get_3x4_RT_matrix_camera_to_world(cam):
    # 获取相机的世界变换矩阵
    location, rotation = cam.matrix_world.decompose()[0:2]
    
    # 转置旋转矩阵以从世界坐标系转换到相机坐标系
    R_bcam2world = rotation.to_matrix()
    print('R_bcam2world: ',R_bcam2world)
    T_bcam2world = location
    print('T_bcam2world: ', T_bcam2world)
    # 将旋转和平移矩阵组合成3x4矩阵
    RT = Matrix((
        R_bcam2world[0][:] + (T_bcam2world[0],),
        R_bcam2world[1][:] + (T_bcam2world[1],),
        R_bcam2world[2][:] + (T_bcam2world[2],)
    ))
    
    return RT

def getCameraIntrinsics (camd):
    fov = camd.angle_y
    scene = bpy.context.scene
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    aspect = w/h
    return fov, aspect
def save_images(save_path_dir,object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
  
    reset_scene()

    # load the object

    load_object(object_file)
    # object_uid = os.path.basename(object_file).split(".")[0]+"_"
    # print(object_uid)

   # normalize_scene(object_file)

    # # create an empty object to track
    # empty = bpy.data.objects.new("Empty", None)
    # scene.collection.objects.link(empty)
    # cam_constraint.target = empty

    randomize_lighting()
   
    # set camera
    for i in range(len(CAMERA_POSE)):

        camera, x, y, z = set_camera_location(cam_pose=CAMERA_POSE[i], i=i)
        # # 获取相机的内参数据
        # fov, aspect = getCameraIntrinsics(camera.data)
        # camera_data = camera.data
        # print('clip_end', camera_data.clip_end)
        # K = get_calibration_matrix_K_from_blender(camera_data)
        # print(K)
            #     # # 保存内参矩阵到CSV文件

        # # 计算内参矩阵
        # K = computeIntrinsicsMatrix(fov, aspect, width=scene.render.resolution_x, height=scene.render.resolution_y)


        # came = getCameraIntrinsics(camd=camera.data)
        # print(came)
        # return
        #print("Rendered image at angle:", angle)
       # camera,x,y,z = randomize_camera(azis=azis,i=i)

        # render the image
        render_file_path = os.path.join(save_path_dir, f"{i:03d}_{x}_{y}_{z}")
        scene.render.filepath = render_file_path
        depth_file_output.file_slots[0].path = render_file_path + "_depth"
        normal_file_output.file_slots[0].path = render_file_path + "_normal"
        albedo_file_output.file_slots[0].path = render_file_path + "_albedo"
        id_file_output.file_slots[0].path = render_file_path + "_id"
        bpy.ops.render.render(write_still=True)

        # save camera RT matrix
        RT = get_3x4_RT_matrix_camera_to_world(camera)
        RT_path = os.path.join(save_path_dir, f"{i:03d}_{x}_{y}_{z}.npy")
        np.save(RT_path, RT)

        # # # 保存内参矩阵到CSV文件
        # csv_filename = os.path.join(render_file_path+'_camera.npy')
        # saveMatrixToCSV(K, csv_filename)



if __name__ == "__main__":
    for name in sorted(os.listdir(INPUT_DIR)):
        input_path_dir = os.path.join(INPUT_DIR, name)
        save_path_dir = os.path.join(OUTPUT_DIR,name)
        if not os.path.exists(save_path_dir):
            os.mkdir(save_path_dir)
        input_obj_path = os.path.join(input_path_dir,'normalized_textured.obj')

        try:
            start_i = time.time()
            save_images(save_path_dir,input_obj_path)
            end_i = time.time()

        except Exception as e:
            print("Failed to render", input_obj_path)
            print(e)