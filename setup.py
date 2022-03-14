from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import numpy
from Cython.Build import cythonize

class build_numpy(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

def install(gmp):
    # -------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------
    # ENTER HERE THE INFORMATION RELATIVE TO CPLEX SUPPORT
    compile_with_cplex = False # 1) Set to True if you want to compil with CPLEX and enable CPLEX support for pruning
    # Path to installation
    CPLEX_BASE_DIR = '/opt/ibm/ILOG/CPLEX_Studio201' # 2) Enter here the path to your CPLEX installation
    # -------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------

    description = 'FairCORELSV2, an enhanced version of the FairCORELS algorithm to build fair optimal rule lists'
    long_description = description
    with open('faircorelsv2/README.md') as f:
        long_description = f.read()

    version = '1.1'

    pyx_file = 'faircorelsv2/_corels.pyx'

    # CPLEX C++ 
    CPLEX_DIR = '%s/cplex' %CPLEX_BASE_DIR
    CPLEX_DIR_CONCERT = '%s/concert' %CPLEX_BASE_DIR

    # Compiler: include directory for both CPLEX C++ API and Concert C++ API
    CXXFLAGS = '-I%s/include' %CPLEX_DIR
    CXXFLAGS2 = '-I%s/include' %CPLEX_DIR_CONCERT
    CPLEX_ADDITIONAL_DIR = '-DIL_STD'
    
    # Linker: specify the CPLEX and Concert libraries
    LINKERDIR_CPLEX = '-L%s/lib/x86-64_linux/static_pic' %CPLEX_DIR
    LINKERDIR_CONCERT = '-L%s/lib/x86-64_linux/static_pic' %CPLEX_DIR_CONCERT
    LDLIBS =  '-lconcert -lilocplex -lcplex -lm -lpthread -ldl'#'-lilocplex -lconcert -lcplex -lm -lpthread'
    #LDFLAGS = '-L%s/lib/x86-64_linux/static_pic' %CPLEX_DIR
    OTHER_ARGS1='-m64'
    OTHER_ARGS2= '-fno-strict-aliasing'
    OTHER_ARGS3= '-fexceptions'
    source_dir = 'faircorelsv2/src/corels/src/'
    sources = ['utils.cpp', 'rulelib.cpp', 'run.cpp', 'pmap.cpp', 
               'corels.cpp', 'cache.cpp',
               # files under this line are for improved filtering only
               'filtering_algorithms.cpp', 
               "milp_pruning_cplex.cpp",
               'mistral_backtrack.cpp', 'mistral_constraint.cpp',
               'mistral_global.cpp', 'mistral_sat.cpp', 'mistral_search.cpp',
               'mistral_solver.cpp', 'mistral_structure.cpp', 'mistral_variable.cpp'
               ]
    
    for i in range(len(sources)):
        sources[i] = source_dir + sources[i]
    
    sources.append(pyx_file)

    sources.append('faircorelsv2/src/utils.cpp')
    #CXXFLAGS, CXXFLAGS2, 
    cpp_args = ['-Wall', '-O3', '-std=c++11', OTHER_ARGS1, OTHER_ARGS2, OTHER_ARGS3]
    if compile_with_cplex:
        cpp_args.extend(['-D CPLEX_SUPPORT', CPLEX_ADDITIONAL_DIR, LINKERDIR_CPLEX, LINKERDIR_CONCERT, LDLIBS])

    libraries = []

    if os.name == 'posix':
        libraries.append('m')

    os.environ["CC"]="/usr/bin/g++"
    os.environ["CXX"]="/usr/bin/g++"

    if gmp:
        libraries.append('gmp')
        cpp_args.append('-DGMP')

    if compile_with_cplex:
        libraries.append('concert')
        libraries.append('ilocplex')
        libraries.append('cplex')
    libraries.append('dl')

    libraries_dirs = []
    if compile_with_cplex:
        libraries_dirs.append('%s/lib/x86-64_linux/static_pic'%CPLEX_DIR)
        libraries_dirs.append('%s/lib/x86-64_linux/static_pic'%CPLEX_DIR_CONCERT)

    if os.name == 'nt':
        cpp_args.append('-D_hypot=hypot')
        if sys.version_info[0] < 3:
            raise Exception("Python 3.x is required on Windows")

    include_dirs_list = ['faircorelsv2/src/', 'faircorelsv2/src/corels/src', numpy.get_include()]

    if compile_with_cplex:
        include_dirs_list.extend(['%s/include'%CPLEX_DIR, '%s/include' %CPLEX_DIR_CONCERT])

    extension = Extension("faircorelsv2._corels", 
                sources = sources,
                libraries = libraries,
                library_dirs = libraries_dirs,
                include_dirs = include_dirs_list,
                language = "c++",
                extra_compile_args = cpp_args)

    extensions = [extension]
    extensions = cythonize(extensions)

    numpy_version = 'numpy'

    if sys.version_info[0] < 3 or sys.version_info[1] < 5:
        numpy_version = 'numpy<=1.16'

    setup(
        name = 'faircorelsv2',
        packages = ['faircorelsv2'],
        ext_modules = extensions,
        version = version,
        author = ' Julien Ferry, Ulrich Aivodji, Sebastien Gambs, Marie-Jose Huguet, Mohamed Siala',
        author_email = 'julienferry12@gmail.com',
        description = description,
        long_description = long_description,
        long_description_content_type='text/markdown',
        setup_requires = [numpy_version],
        install_requires = [numpy_version],
        python_requires = '>=2.7',
        url = 'https://github.com/ferryjul/fairCORELSV2',
        #download_url = 'https://github.com/ferryjul/fairCORELSV2/archive/0.7.tar.gz',
        #cmdclass = {'build_ext': build_numpy},
        license = "GNU General Public License v3 (GPLv3)",
        classifiers = [
            "Programming Language :: C++",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent"
        ]
    )

if __name__ == "__main__":
    try:
        install(True)
    except:
        install(False)
