#!/usr/bin/env python3
"""
Setup script for Drone Simulation System
Configures the package for installation and distribution
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Drone simulation and control system with 3D visualization"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
    try:
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return [
            'numpy>=1.21.0',
            'PyYAML>=6.0',
            'pygame>=2.0.0',
            'PyOpenGL>=3.1.0',
            'PyOpenGL_accelerate>=3.1.0',
            'scipy>=1.7.0',
            'matplotlib>=3.5.0'
        ]

setup(
    name='drone-simulator',
    version='1.0.0',
    description='Advanced drone simulation and control system with 3D visualization',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Drone Project Team',
    author_email='drone@project.com',
    url='https://github.com/drone-project/simulator',
    
    # Package configuration
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies for development
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.800'
        ],
        'analysis': [
            'jupyter>=1.0',
            'pandas>=1.3.0',
            'seaborn>=0.11.0'
        ]
    },
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'drone-simulator=simulator:main',
            'drone-real=real:main',
            'drone-tests=run_tests:main'
        ]
    },
    
    # Package data
    package_data={
        'cfg': ['*.yaml', '*.json'],
        'docs': ['*.md', '*.txt'],
        '': ['*.yaml', '*.json', '*.md']
    },
    
    # Classification
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Games/Entertainment :: Simulation',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent'
    ],
    
    # Keywords for discovery
    keywords='drone simulation physics 3d opengl pygame control ai robotics',
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/drone-project/simulator/issues',
        'Source': 'https://github.com/drone-project/simulator',
        'Documentation': 'https://drone-project.readthedocs.io'
    },
    
    # Zip safe
    zip_safe=False,
    
    # Test suite
    test_suite='tests',
    tests_require=['pytest>=6.0']
)
