from setuptools import setup

setup(
    name='libscf',
    version='0.0.1',
    py_modules=['pyscf'],
    install_requires=[
        'numpy',"open3d","pyshtools","pytransform3d",
    ],
)