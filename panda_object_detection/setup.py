from setuptools import find_packages, setup

package_name = 'panda_object_detection'

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
    maintainer='manoj',
    maintainer_email='manojkumareedhara3@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection_node = panda_object_detection.object_detection_node:main',
            'pick_and_place = panda_object_detection.pick_and_place:main',
            'gripper_object_monitor = panda_object_detection.gripper_object_monitor:main',
        ],
    },
)
