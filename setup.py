
from setuptools import setup
setup(name='Housing',
      version='1.0',
      description='Practice ML & travis/coveralls with housing prices data set',
      long_description='',
      author='Jaron C Whittington',
      author_email='jaronwhitty@gmail.com',
      url='https://github.com/JaronWhitty/Medic',
      license='MIT',
      setup_requires=['pytest-runner',],
      tests_require=['pytest', 'python-coveralls', 'coverage'],
      install_requires=[
          "pandas",
          "sklearn",
          "numpy",
          "scipy"
      ],
      packages = ['housing_predictor'],
      include_package_data=True,
      scripts=['housing_predictor/housing_predictor.py'],
              
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Other Audience',
          'Natural Language :: English',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6'
      ],
)
