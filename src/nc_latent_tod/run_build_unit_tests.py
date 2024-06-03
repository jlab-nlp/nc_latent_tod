import os
import unittest

from nc_latent_tod.utils.testing import load_suite


if __name__ == '__main__':
    repo_root_directory = os.environ.get('GITHUB_WORKSPACE', os.getcwd())
    suite = load_suite('unit_build', directory=os.path.join(repo_root_directory, 'src', 'nc_latent_tod'))
    print("Suite loaded", suite)
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if not result.wasSuccessful():
        result.printErrors()
        raise RuntimeError("Unit tests failed")
