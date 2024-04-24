from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
import pandas as pd

import copy
from collections import *

class abc_setting(ABC):    
    @abstractmethod
    def __init__(self, setting='abstract_setting'):
        self.setting = setting
    
    @abstractmethod
    def e_function(self):
        pass
    
    @abstractmethod
    def get_suff_stat(self, idx, **kwargs):
        """
        Returns the sufficient statistic for hypothesis H_j given the data.
        """
    
    @abstractmethod
    def cond_e_sampling(self, idx, **kwargs):
        """
        Conditionally samples from the joint distribution (e_1, ..., e_m) | S_j under the null H_j.
        """
        pass
    
    def get_setting(self):
        return (self.setting)
    
    def get_attributes(self):
        return vars(self)