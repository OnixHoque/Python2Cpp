from setuptools import setup, Extension

setup(name='funniest',
      version='0.1',
      description='The funniest joke in the world',
      url='http://github.com/storborg/funniest',
      author='Flying Circus',
      author_email='flyingcircus@example.com',
      license='MIT',
      packages=['funniest'],
	ext_modules=[Extension('funniest.foo', ['csrc/foo.c'])],	
      zip_safe=False)
