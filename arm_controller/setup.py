from setuptools import find_packages, setup

package_name = 'arm_controller'

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
    maintainer='atharva23',
    maintainer_email='atharvad2366@gmail.com',
    description='Arm controller: aruco, bad-fruit perception and manipulation nodes',
    license='ASR Robotics',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'aruco_perception = arm_controller.aruco_perception:main',
            'bad_fruits_perception = arm_controller.bad_fruits_perception:main',
            'arm_manipulation = arm_controller.arm_manipulation:main',
        ],
    },
)
