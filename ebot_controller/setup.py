from setuptools import find_packages, setup

package_name = 'ebot_controller'

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
    description='eBot package containing navigation and LiDAR shape detection nodes',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ebot_nav = ebot_controller.ebot_nav:main',
            'shape_detector = ebot_controller.shape_detector:main',
        ],
    },
)
