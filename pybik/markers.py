#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 9 14:13:21 2019

@author: todd
"""


from copy import deepcopy
import numpy as np



class Marker(object):
    def __init__(self, q):
        self.q = q


class MarkerCluster(object):
    
    def __init__(self, q, label=None):
        self.label   = label
        self.markers = [Marker(qq) for qq in q]
        self.noise   = np.zeros( self.shape )

    def __repr__(self):
        s  = 'Markers\n' if (self.label is None) else 'Markers (%s)\n' %self.label
        s += '    n        = %d\n' %self.n
        s += '    centroid = %s\n' %str( self.centroid )
        return s
    
    @property
    def centroid(self):
        return self.toarray().mean(axis = 0)
    
    @property
    def shape(self):
        return (self.n, 3)

    @property
    def n(self):
        return len( self.markers )
    
    def add_noise(self, sigma = 1, newobj = True, label = None):
        if newobj:
            m = self.copy()
            m.add_noise(sigma, newobj = False, label = label)
            return m
        else:
            self.noise = np.random.normal(0, sigma, self.shape)
            self.set_label(label)

    def copy(self):
        return deepcopy(self)

    def new(self, q, label=None):
        return MarkerCluster(q, label)
    
    def remove_noise(self):
        self.noise   = np.zeros( self.shape )
    
    def reset(self):
        self.remove_noise()
    
    def set_label(self, label):
        self.label = label
    
    def toarray(self, centered = False):
        q      = np.array([m.q for m in self.markers])
        q     += self.noise
        if centered:
            q -= q.mean(axis = 0)
        return q
    
    def noisemagnitude(self):
        return np.linalg.norm(self.noise)
    
    
    
    
    
    
    
    
    
