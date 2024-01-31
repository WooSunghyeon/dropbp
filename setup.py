from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name='dropbp',
      ext_modules=[
          cpp_extension.CUDAExtension(
              'dropbp.cpp_extention.allocate_p',
              ['dropbp/cpp_extention/allocate_p.cc']
          ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
      )