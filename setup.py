from setuptools import setup, find_packages

setup(
  name = 'dalle2-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'DALL-E 2',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/dalle2-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'text to image'
  ],
  install_requires=[
    'einops>=0.4',
    'torch>=1.6',
    'x-clip>=0.4.1'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
