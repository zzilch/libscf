# Install
```
pip install .
```

# Usage
```
import trimesh
import pyvista as pv
from pyvista import examples
import numpy as np
import matplotlib.pyplot as plt
from pyscf import compute_spfs,normalize_sph,batch_sph2scf

mesh = examples.download_teapot()
mesh = trimesh.Trimesh(mesh.points,mesh.faces.reshape(-1,4)[:,1:])
mesh = mesh.apply_scale(1/mesh.scale)
points = mesh.bounding_sphere.to_mesh().bounding_box.sample_grid(4)

spfs,rays = compute_spfs(mesh,points,return_rays=True)
scfs = batch_sph2scf(spfs)
plt.imshow(scfs[:,:5])
plt.show()

pl = pv.Plotter()
pl.add_mesh(mesh,opacity=0.5)
for i in range(len(points)):
    sp = rays[i][:,:3]+rays[i][:,3:]*0.05
    spf = spfs[i]
    spf_norm = normalize_sph(spf)
    pl.add_mesh(sp,scalars=spf_norm)
pl.show()
```