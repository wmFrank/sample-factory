import setuptools
from setuptools import setup


with open('README.md', 'r') as f:
    long_description = f.read()
    descr_lines = long_description.split('\n')
    descr_no_gifs = []  # gifs are not supported on PyPI web page
    for dl in descr_lines:
        if not ('<img src=' in dl and 'gif' in dl):
            descr_no_gifs.append(dl)

    long_description = '\n'.join(descr_no_gifs)


setup(
    # Information
    name='sample-factory',
    description='High throughput asynchronous reinforcement learning framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='2.0.0',
    url='https://github.com/alex-petrenko/sample-factory',
    author='Aleksei Petrenko',
    license='MIT',
    keywords='asynchronous reinforcement learning policy gradient ppo appo impala ai',
    project_urls={
        'Github': 'https://github.com/alex-petrenko/sample-factory',
        'Videos': 'https://sites.google.com/view/sample-factory',
    },

    install_requires=[
        'numpy>=1.18.1<2.0',
        'torch>=1.6,<1.12',
        'gym>=0.17.1,<=0.21.0',
        'pyglet',  # gym dependency
        'tensorboard>=1.15.0',
        'tensorboardx>=2.0',
        'psutil>=5.7.0',
        'threadpoolctl>=2.0.0',
        'colorlog',
        'faster-fifo>=1.4.0,<2.0',
        'filelock',
        'opencv-python',
        'wandb>=0.12.9',
    ],
    extras_require={
        'dev': ['black', 'isort', 'pytest<8.0', 'flake8'],
        'atari': ['gym[atari, accept-rom-license]'],
        'mujoco': ['gym[mujoco]<1.0', 'mujoco', 'mujoco_py>=2.0,<3.0'],
        'vizdoom': ['vizdoom<2.0'],
    },

    package_dir={'': './'},
    packages=setuptools.find_packages(where='./', include='sample_factory*'),
    include_package_data=True,

    python_requires='>=3.7',
)
