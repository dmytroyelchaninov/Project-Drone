from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="drone-sim",
    version="0.1.0",
    author="Drone Simulation Team",
    author_email="info@dronesim.com",
    description="A physics-accurate, reactive simulator for multirotor drones with acoustic analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/drone-sim",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.950",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "PyOpenGL>=3.1.0",
            "pygame>=2.1.0",
        ],
        "ml": [
            "torch>=1.12.0",
            "gym>=0.26.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "drone-sim=drone_sim.ui.cli:main",
        ],
    },
) 