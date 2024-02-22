from setuptools import find_packages, setup

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

if __name__ == "__main__":
    setup(
        # Needed to silence warnings (and to be a worthwhile package)
        name='nonlinear-causal',
        url='https://github.com/statmlben/nonlinear-causal',
        author='Ben Dai',
        author_email='bendai@cuhk.edu.hk',
        # Needed to actually package something
        packages=find_packages(),
        # Needed for dependencies
        install_requires=['numpy==1.22', 'pandas', 'sliced', 'scipy', 'scikit-learn', 'statsmodels'],
        # *strongly* suggested for sharing
        version='1.0',
        # The license can be anything you like
        license='MIT',
        description='nonlinear-causal is a Python module for nonlinear causal inference built on top of Two-stage IV methods.',
        #cmdclass={"build_ext": build_ext},
        # We will also need a readme eventually (there will be a warning)
        long_description_content_type='text/markdown',
        long_description=LONG_DESCRIPTION,
    )
