from setuptools import find_packages, setup

package_name = 'high_level_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='x',
    maintainer_email='x.x@x.x',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'control_gripper2 = high_level_control.control_gripper2:main',
        'config_avoid_wall_1p = high_level_control.config_avoid_wall_1p:main',
        'control_avoid_wall_3p = high_level_control.control_avoid_wall_3p:main',
        ],
    },
)
