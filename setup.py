from setuptools import setup

setup(
    name='lcmpc',
    version='0.1.0',    
    description='Loop Closure MPC with CasADi',
    url='https://github.com/kotakondo/loop-closure-mpc',
    author='Kota Kondo',
    author_email='kkondo@mit.edu',
    license='MIT',
    packages=['lcmpc'],
    install_requires=['numpy',
                        'matplotlib',
                        'ipython',
                        'sympy',
                        'scipy',
                        'casadi',
                        'ipywidgets',
                      ],

    # classifiers=[
    #     'Development Status :: 1 - Planning',
    #     'Intended Audience :: Science/Research',
    #     'License :: OSI Approved :: BSD License',  
    #     'Operating System :: POSIX :: Linux',        
    #     'Programming Language :: Python :: 2',
    #     'Programming Language :: Python :: 2.7',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.4',
    #     'Programming Language :: Python :: 3.5',
    # ],
)