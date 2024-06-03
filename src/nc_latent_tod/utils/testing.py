import os
import unittest
from typing import Set, Tuple


def test_suite(tag):
    """
    Allows tagging of test suites for selective running. Just put @test_suite("suite_name") above the class definition.
    """
    def decorator(cls):
        cls._test_suite_tag = tag
        return cls
    return decorator


def load_suite(tag, directory: str = None):
    directory = directory or os.getcwd()
    print("Loading tests from", directory)
    print("Current Contents:", "\n".join(os.listdir(directory)))
    tag_suite = unittest.TestSuite()
    added_tests: Set[Tuple[str, str]] = set()
    for all_test_suites in unittest.TestLoader().discover(directory):
        for suite in all_test_suites:
            # Check if suite is actually a test suite and not a failed test
            if isinstance(suite, unittest.TestSuite):
                for test in suite:
                    # Check if test is an instance of a TestCase
                    if isinstance(test, unittest.TestCase):
                        # Check for the tag attribute
                        if hasattr(test.__class__, '_test_suite_tag') and getattr(test.__class__, '_test_suite_tag') == tag:
                            test_identifier: Tuple[str, str] = (test.__class__.__module__, test.__class__.__name__)
                            if test_identifier not in added_tests:
                                # This prevents us from loading duplicate tests, since our tag is at the class level
                                added_tests.add(test_identifier)
                                tag_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(test.__class__))
                    elif isinstance(test, unittest.TestSuite):
                        # Handle nested test suites
                        # You can implement similar logic here if needed
                        pass
            elif isinstance(suite, unittest.TestCase):
                # Handle individual test cases
                # Implement logic here if needed
                pass
    return tag_suite
