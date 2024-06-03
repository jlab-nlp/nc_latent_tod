import unittest

from nc_latent_tod.mwzeval.normalization import time_str_to_minutes

"""
Testing normalization in mwzeval, centered arround any modifications made for this work
"""


class MWZEvalNormalizationTests(unittest.TestCase):
    def test_time_str_to_minutes(self):
        # testing valid inputs
        self.assertEqual(time_str_to_minutes("09:30"), (9 * 60) + 30)
        self.assertEqual(time_str_to_minutes("9:30"), (9 * 60) + 30)
        self.assertEqual(time_str_to_minutes("7:41"), (7 * 60) + 41)
        self.assertEqual(time_str_to_minutes("13:02"), (13 * 60) + 2)
        self.assertEqual(time_str_to_minutes("23:59"), (23 * 60) + 59)
        self.assertEqual(time_str_to_minutes("00:00"), 0)
        self.assertEqual(time_str_to_minutes("0:00"), 0)

        # testing invalid inputs
        self.assertEqual(time_str_to_minutes("10:15onsaturday"), 0)
        self.assertEqual(time_str_to_minutes("somethingbefore10:15"), 0)



if __name__ == '__main__':
    unittest.main()
