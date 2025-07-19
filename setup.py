from setuptools import setup, find_packages

# Read requirements from requirements.txt to avoid duplication
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pair_trading_system',
    version='1.0.0',
    description='An integrated system for researching, backtesting, and deploying pairs trading strategies.',
    author='Arnav Jain',
    author_email='ajain.careers@gmail.com',
    url='https://github.com/AJAllProjects/Pairs_Trading_System',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    python_requires='>=3.9',
)