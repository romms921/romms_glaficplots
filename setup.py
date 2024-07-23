from setuptools import setup

setup(
    name='romms_glaficplots',
    version='0.0.29',    
    description='A python package to make basic plots (errors and critical curves) for glafic',
    url='https://github.com/romms921/romms_glaficplots.git',
    author='Rommulus Lewis',
    author_email='rommuluslewis@gmail.com',
    license='BSD 2-clause',
    keywords=['astronomy','astrophysics','gravitationallensing', 'lensing', 'glafic'], 
    packages=['romms_glaficplots'],
    install_requires=['pandas',
                      'numpy',
                      'matplotlib',
                      'seaborn',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)