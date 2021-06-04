from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='mce_irl_pomdps',
   version='0.1',
   description='A module for finding optimal policy and learning rewards'+\
                    ' for iverse reinforcement learning problems'+\
                    ' in partially observable environments',
   license="GNU 3.0",
   long_description=long_description,
   packages=['mce_irl_pomdps'],
   package_dir={'mce_irl_pomdps': 'mce_irl_pomdps/'},
   install_requires=['numpy', 'matplotlib'],
)