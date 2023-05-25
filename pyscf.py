import numpy as np
import open3d as o3d
import pyshtools
from pytransform3d import coordinates,rotations
from functools import partial

INVALID_ID = o3d.t.geometry.RaycastingScene.INVALID_ID

def create_raycast_scene(mesh):
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices.astype('float32')),
        o3d.utility.Vector3iVector(mesh.faces.astype('int32')))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d))
    return scene

def closest_point(scene,points):
    points_o3dt = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    ans = scene.compute_closest_points(points_o3dt)
    closest = ans['points'].numpy().astype('float32')
    distance = np.linalg.norm(points-closest,axis=-1)
    triangle_id = ans['primitive_ids'].numpy().astype('int32')
    return closest,distance,triangle_id

def cast_rays(scene,rays):
    rays_o3dt = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    hits = scene.cast_rays(rays_o3dt)
    distance = hits['t_hit'].numpy()
    return distance.astype('float32')

def create_rays_base(L=15):
    grid = pyshtools.SHGrid.from_zeros(lmax=L)
    lats = grid.lats(degrees=False)+np.pi/2 # (-pi,pi)->(0,pi)
    lons = grid.lons(degrees=False)-np.pi   # (0,2*pi)->(-pi,pi)
    uv = np.stack(np.meshgrid(lats,lons,indexing='ij'),axis=-1).reshape(-1,2)
    ruv = np.concatenate([np.ones_like(uv[:,:1]),uv],axis=-1)
    xyz = coordinates.cartesian_from_spherical(ruv)
    return xyz

def batch_rotor_apply(q, v):
    t = 2 * np.cross(q[...,1:], v)
    return v + q[...,:1] * t + np.cross(q[...,1:], t)

def create_rays(rays_base,center,dir):
    rotor = rotations.rotor_from_two_directions(dir,rotations.unitz)
    dirs = batch_rotor_apply(rotor,rays_base)
    rays = np.concatenate((
        center+np.zeros_like(dirs),
        dirs
    ),axis=-1).astype('float32')
    return rays

def normalize_sph(spf):
    mask = np.isinf(spf)
    if np.all(mask): 
        spf = np.zeros_like(spf)
    else:
        spf_valid = spf[~mask]
        spf_mean = spf_valid.mean()
        spf_min = spf_valid.min()
        spf = (spf_min+spf_mean)/(spf+spf_mean)
    return spf

def sph2scf(spf,L=15):
    shape = [2*L+2+1,(2*L+2)*2+1]
    spf = normalize_sph(spf).reshape(shape)
    scf = pyshtools.SHGrid.from_array(spf).expand().spectrum()
    return scf

def compute_spfs(mesh,points,L=15,return_rays=False):
    scene = create_raycast_scene(mesh)
    nn,d,_ = closest_point(scene,points)
    dirs = (nn-points)/d[...,None]
    rays_base = create_rays_base(L=L)
    rays = list(map(partial(create_rays,rays_base),points,dirs))
    spfs = cast_rays(scene,rays).reshape(len(points),-1)
    if return_rays:
        return spfs,np.stack(rays,0)
    return spfs

def batch_sph2scf(spfs,L=15):
    scfs = np.apply_along_axis(sph2scf,1,spfs,L).astype('float32')
    return scfs

def batch_compute_scfs(mesh,points,L=15):
    spfs = compute_spfs(mesh,points,L=L)
    scfs = np.apply_along_axis(sph2scf,1,spfs,L).astype('float32')
    return scfs