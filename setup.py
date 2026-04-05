from setuptools import find_packages, setup

package_name = 'quad_trajectories'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'jax', 'jaxlib'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Standalone trajectory definitions for quadrotor control',
    license='MIT',
)
