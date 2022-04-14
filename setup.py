from setuptools import find_packages, setup

setup(
    name="dalle2-pytorch",
    packages=find_packages(exclude=[]),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "dalle2_pytorch = dalle2_pytorch.cli:main",
            "dream = dalle2_pytorch.cli:dream",
        ],
    },
    version="0.0.15",
    license="MIT",
    description="DALL-E 2",
    author="Phil Wang",
    author_email="lucidrains@gmail.com",
    url="https://github.com/lucidrains/dalle2-pytorch",
    keywords=["artificial intelligence", "deep learning", "text to image"],
    install_requires=[
        "click",
        "einops>=0.4",
        "einops-exts>=0.0.3",
        "kornia>=0.5.4",
        "pillow",
        "torch>=1.10",
        "torchvision",
        "tqdm",
        "x-clip>=0.4.4",
        "youtokentome",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
