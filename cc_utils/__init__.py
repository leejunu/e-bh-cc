# this submodule is for objects that characterize the distributional assumptions of the testing problem (e.g., z-testing).
# each object will contain methods that allow e-functions, p-functions, sufficient statistic functions, and sampling
# for the CC mechanism, which requires the practictioner to sample from, for each index j, the joint distribution of
# e-values given a sufficient statistic S_j under the null hypothesis H_j.

from .abstract_test_setting import abc_setting

from .ztesting import ztesting
from .ttesting import ttesting

# from .mxknockoffs import knockoffs, fxknockoffs, gaussian_mxknockoffs
# have to load in knockoffs separately because of knockpy dependency

# from .conformal import conformal
# have to load in conformal separately because of numba dependency
