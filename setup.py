import os
from setuptools import setup, find_packages

def read_environment_yaml():
    """Read dependencies from environment.yaml"""
    env_file = 'environment.yaml'
    if not os.path.exists(env_file):
        return []
    
    try:
        import yaml
        with open(env_file, 'r') as f:
            env_data = yaml.safe_load(f)
    except ImportError:
        # PyYAML not available, fallback to basic parsing
        return []
    
    # Extract pip dependencies
    pip_deps = []
    dependencies = env_data.get('dependencies', [])
    
    for dep in dependencies:
        if isinstance(dep, dict) and 'pip' in dep:
            pip_deps.extend(dep['pip'])
    
    # Filter out packages that are typically conda-only or not needed for setup.py
    essential_deps = []
    # for dep in pip_deps:
    #     pkg_name = dep.split('==')[0].split('>=')[0].split('<=')[0]
    #     # Keep only essential dependencies for the core package
    #     if pkg_name.lower() in ['numpy', 'matplotlib', 'scikit-learn', 'scikit-image', 'pot', 'tqdm']:
    #         essential_deps.append(dep)
    
    return essential_deps

def get_base_requirements():
    """Get base requirements that are always needed"""
    return [
        'torch',
        'numpy',
    ]

# Read all dependencies from environment.yaml
try:
    env_requirements = read_environment_yaml()
    install_requires = get_base_requirements() + env_requirements
except Exception:
    install_requires = get_base_requirements()

setup(
    name='partial-tsw',
    version='0.1.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=install_requires,
    author='Thanh Tran',
    author_email='thanhqt2002@gmail.com',
    description='Tree-Sliced Entropy Partial Transport',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thanhqt2002/PartialTSW',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)