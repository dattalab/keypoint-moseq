import setuptools

with open("README.md", 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='keypoint-moseq',
    version='0.0.4',
    author='Caleb Weinreb',
    author_email='calebsw@gmail.com',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.8',
    install_requires=[
        'seaborn',
        'cytoolz',
        'matplotlib',
        'tqdm',
        'ipykernel',
        'imageio[ffmpeg]',
        'statsmodels',
        'pyyaml',
        'vidio',
        'holoviews[recommended]',
        'bokeh',
        'pandas',
        'tables',
        'jax-moseq',
    ], 
    url='https://github.com/dattalab/keypoint-moseq'
)
