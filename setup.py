from setuptools import setup, find_packages

setup(
    name='mask-detection-system-1',
    version='dev_v1.0',
    description='mask detection',
    keywords='mask',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        # 而cfg并不是一个包，因此需要指定cfg包名
        "cfg": ["./*.cfg"],
    },
    scripts=["./detect.py",
             "./models.py",
             "./test.py",
             "./train.py",
             ],
    install_requires=[
        'numpy',
        'opencv-python-headless',
        'torch',
        'matplotlib',
        'pycocotools',
        'tqdm',
        'pillow',
        'pydub',
        'pyttsx3'
    ],
        author='CreateLAB',
)
