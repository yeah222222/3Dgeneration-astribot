"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
import uuid
from typing import Tuple
from mathutils import Vector, Matrix
import numpy as np

import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="/home/ec2-user/zjc_renderobjaverse/test_output2")#  ~/.objaverse/hf-objaverse-v1/views_whole_sphere
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--num_images", type=int, default=8)
parser.add_argument("--camera_dist", type=int, default=1.2)
    
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print('===================', args.engine, '===================')
FORMAT = 'OPEN_EXR'
COLOR_DEPTH = '16'
context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

# setup lighting
bpy.ops.object.light_add(type="AREA")
light2 = bpy.data.lights["Area"]
light2.energy = 3000
bpy.data.objects["Area"].location[2] = 0.5
bpy.data.objects["Area"].scale[0] = 100
bpy.data.objects["Area"].scale[1] = 100
bpy.data.objects["Area"].scale[2] = 100

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 256
render.resolution_y = 256
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = "CUDA" # or "OPENCL"

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
# else:
#     depth_file_output.format.color_mode = "BW"

#     # Remap as other types can not represent the full range of depth.
#     map = nodes.new(type="CompositorNodeMapValue")
#     # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
#     map.offset = [-0.7]
#     map.size = [DEPTH_SCALE]
#     map.use_min = True
#     map.min = [0]

#     links.new(render_layers.outputs['Depth'], map.inputs[0])
#     links.new(map.outputs[0], depth_file_output.inputs[0])

def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )

def sample_spherical(radius=3.0, maxz=3.0, minz=0.):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        vec[2] = np.abs(vec[2])
        vec = vec / np.linalg.norm(vec, axis=0) * radius
        if maxz > vec[2] > minz:
            correct = True
    return vec

def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
#         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec
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
def get_calibration_matrix_K_from_blender(camd):
    # print(camd.data.type)
    if camd.data.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.data.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.data.sensor_fit, camd.data.sensor_width, camd.data.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.data.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.data.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.data.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K
def randomize_camera():
    elevation = random.uniform(0., 90.)
    azimuth = random.uniform(0., 360)
    distance = random.uniform(0.8, 1.6)
    return set_camera_location(elevation, azimuth, distance)

def set_camera_location(elevation, azimuth, distance):
    # from https://blender.stackexchange.com/questions/18530/
    x, y, z = sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera,x,y,z

def randomize_lighting() -> None:
    light2.energy = random.uniform(300, 600)
    bpy.data.objects["Area"].location[0] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[1] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[2] = random.uniform(0.5, 1.5)


def reset_lighting() -> None:
    light2.energy = 1000
    bpy.data.objects["Area"].location[0] = 0
    bpy.data.objects["Area"].location[1] = 0
    bpy.data.objects["Area"].location[2] = 0.5


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


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
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


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
# def get_3x4_RT_matrix_from_blender(cam):
#     # bcam stands for blender camera
#     # R_bcam2cv = Matrix(
#     #     ((1, 0,  0),
#     #     (0, 1, 0),
#     #     (0, 0, 1)))

#     # Transpose since the rotation is object rotation, 
#     # and we want coordinate rotation
#     # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
#     # T_world2bcam = -1*R_world2bcam @ location
#     #
#     # Use matrix_world instead to account for all constraints
#     location, rotation = cam.matrix_world.decompose()[0:2]
#     R_world2bcam = rotation.to_matrix().transposed()

#     # Convert camera location to translation vector used in coordinate changes
#     # T_world2bcam = -1*R_world2bcam @ cam.location
#     # Use location from matrix_world to account for constraints:     
#     T_world2bcam = -1*R_world2bcam @ location

#     # # Build the coordinate transform matrix from world to computer vision camera
#     # R_world2cv = R_bcam2cv@R_world2bcam
#     # T_world2cv = R_bcam2cv@T_world2bcam

#     # put into 3x4 matrix
#     RT = Matrix((
#         R_world2bcam[0][:] + (T_world2bcam[0],),
#         R_world2bcam[1][:] + (T_world2bcam[1],),
#         R_world2bcam[2][:] + (T_world2bcam[2],)
#         ))
#     return RT

def normalize_scene():
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

def get_3x4_RT_matrix_camera_to_world(cam):
    # 获取相机的世界变换矩阵
    location, rotation = cam.matrix_world.decompose()[0:2]
    
    # 转置旋转矩阵以从世界坐标系转换到相机坐标系
    R_bcam2world = rotation.to_matrix()
    # print('R_bcam2world: ',R_bcam2world)
    T_bcam2world = location
    # print('T_bcam2world: ', T_bcam2world)
    # 将旋转和平移矩阵组合成3x4矩阵
    RT = Matrix((
        R_bcam2world[0][:] + (T_bcam2world[0],),
        R_bcam2world[1][:] + (T_bcam2world[1],),
        R_bcam2world[2][:] + (T_bcam2world[2],)
    ))
    
    return RT
def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)

    reset_scene()

    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene()

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    randomize_lighting()
    for i in range(args.num_images):
        # # set the camera position
        # theta = (i / args.num_images) * math.pi * 2
        # phi = math.radians(60)
        # point = (
        #     args.camera_dist * math.sin(phi) * math.cos(theta),
        #     args.camera_dist * math.sin(phi) * math.sin(theta),
        #     args.camera_dist * math.cos(phi),
        # )
        # # reset_lighting()
        # cam.location = point

        # set camera
        camera, x, y, z = randomize_camera()
        K = get_calibration_matrix_K_from_blender(camd=camera)
        render_file_path = os.path.join(args.output_dir,object_uid, f"{i:03d}_{x}_{y}_{z}")
        scene.render.filepath = render_file_path
        depth_file_output.file_slots[0].path = render_file_path + "_depth"
        # render the image
        render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}_{x}_{y}_{z}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

        # save camera RT matrix
        RT = get_3x4_RT_matrix_camera_to_world(camera)
        RT_path = os.path.join(args.output_dir, object_uid, f"{i:03d}_{x}_{y}_{z}.npy")
        np.save(RT_path, RT)
        #save camera K matrix
        
        K_path = os.path.join(args.output_dir, object_uid, f"{i:03d}_{x}_{y}_{z}_K.npy")
        np.save(K_path, K)

def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)