import unittest
import numpy as _np

from dtw_dist import dtw_dist 

class DTWTestSuite(unittest.TestCase):
    
    def test_dtw_dist(self):
        X = _np.array([1, -1, 0, 0, 0])
        Y = _np.array([0, 0, 1, -1, 0])
        dist_0_dep = dtw_dist(X, Y, mode='dependent')
        dist_0_indep = dtw_dist(X, Y, mode='independent')
        dist_1_dep = dtw_dist(X, Y, w=1, mode='dependent')
        dist_1_indep = dtw_dist(X, Y, w=1, mode='independent')
        self.assertAlmostEqual(dist_0_dep, 2.000, places=4)
        self.assertAlmostEqual(dist_0_dep, dist_0_indep, places=4)
        self.assertAlmostEqual(dist_1_dep, 4.000, places=4)
        self.assertAlmostEqual(dist_1_dep, dist_1_indep, places=4)
        
if __name__ == '__main__':
    unittest.main()