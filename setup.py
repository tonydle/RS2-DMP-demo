import os
from glob import glob
from setuptools import setup

package_name = 'rs2_dmp_demo'
submodules = [
    'cdmp',
]

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, *submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tony Le',
    maintainer_email='tonyle98@outlook.com',
    description='Demo for CDMPs',
    license='MIT',
    entry_points={
        'console_scripts': [
            'test_cdmp_panda = rs2_dmp_demo.test_cdmp_panda:main',
            'servo_test = rs2_dmp_demo.servo_test:main',
        ],
    },
)
