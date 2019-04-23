import numpy as _np
import pyximport; pyximport.install(setup_args={'include_dirs': _np.get_include()})
import unittest

from ts_dist import dtw_dist as dtw_dist_py
from ts_dist import lcss_dist as lcss_dist_py
from ts_dist import edr_dist as edr_dist_py

from ts_dist_cy import dtw_dist as dtw_dist_cy
from ts_dist_cy import lcss_dist as lcss_dist_cy
from ts_dist_cy import edr_dist as edr_dist_cy


class TSDISTTestSuite(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = _np.array([1, -1, 0, 0, 0])
        self.Y = _np.array([0, 0, 1, -1, 0])

    def test_dtw_dist_py(self):
        dist_0_dep = dtw_dist_py(self.X, self.Y, mode="dependent")
        dist_0_indep = dtw_dist_py(self.X, self.Y, mode="independent")
        dist_1_dep = dtw_dist_py(self.X, self.Y, w=1, mode="dependent")
        dist_1_indep = dtw_dist_py(self.X, self.Y, w=1, mode="independent")
        self.assertAlmostEqual(dist_0_dep, 2.000, places=4)
        self.assertAlmostEqual(dist_0_dep, dist_0_indep, places=4)
        self.assertAlmostEqual(dist_1_dep, 4.000, places=4)
        self.assertAlmostEqual(dist_1_dep, dist_1_indep, places=4)

    def test_lcss_dist_py(self):
        dist_0 = lcss_dist_py(self.X, self.Y, delta=_np.inf, epsilon=0.5)
        dist_1 = lcss_dist_py(self.X, self.Y, delta=2, epsilon=0.5)
        self.assertAlmostEqual(dist_0, 0.4000, places=4)
        self.assertAlmostEqual(dist_1, 0.6000, places=4)

    def test_edr_dist_py(self):
        dist_0 = edr_dist_py(self.X, self.Y, epsilon=0.5)
        self.assertAlmostEqual(dist_0, 0.8000, places=4)

    def test_dtw_dist_cy(self):
        dist_0_dep = dtw_dist_cy(self.X, self.Y, mode="dependent")
        dist_0_indep = dtw_dist_cy(self.X, self.Y, mode="independent")
        dist_1_dep = dtw_dist_cy(self.X, self.Y, w=1, mode="dependent")
        dist_1_indep = dtw_dist_cy(self.X, self.Y, w=1, mode="independent")
        self.assertAlmostEqual(dist_0_dep, 2.000, places=4)
        self.assertAlmostEqual(dist_0_dep, dist_0_indep, places=4)
        self.assertAlmostEqual(dist_1_dep, 4.000, places=4)
        self.assertAlmostEqual(dist_1_dep, dist_1_indep, places=4)

    def test_lcss_dist_cy(self):
        dist_0 = lcss_dist_cy(self.X, self.Y, delta=_np.inf, epsilon=0.5)
        dist_1 = lcss_dist_cy(self.X, self.Y, delta=2, epsilon=0.5)
        self.assertAlmostEqual(dist_0, 0.4000, places=4)
        self.assertAlmostEqual(dist_1, 0.6000, places=4)

    def test_edr_dist_cy(self):
        dist_0 = edr_dist_cy(self.X, self.Y, epsilon=0.5)
        self.assertAlmostEqual(dist_0, 0.8000, places=4)

if __name__ == "__main__":
    unittest.main()