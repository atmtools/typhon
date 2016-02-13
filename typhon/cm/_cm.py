# -*- coding: utf-8 -*-
"""
Colormap definitions.

Acknowledegments:

The qualitative color palettes are ported versions of ColorBrewer's Set1 and
Set3 and include color specifications and designs developed by Cynthia Brewer
[0].

[0] http://colorbrewer.org/

"""

_qualitative1_data = [
    [ 0.89411765,  0.10196078,  0.10980392],
    [ 0.21568627,  0.49411765,  0.72156863],
    [ 0.30196078,  0.68627451,  0.29019608],
    [ 0.59607843,  0.30588235,  0.63921569],
    [ 1.        ,  0.49803922,  0.        ],
    [ 1.        ,  1.        ,  0.2       ],
    [ 0.65098039,  0.3372549 ,  0.15686275]]

_qualitative2_data = [
    [ 0.55294118,  0.82745098,  0.78039216],
    [ 1.        ,  1.        ,  0.70196078],
    [ 0.74509804,  0.72941176,  0.85490196],
    [ 0.98431373,  0.50196078,  0.44705882],
    [ 0.50196078,  0.69411765,  0.82745098],
    [ 0.99215686,  0.70588235,  0.38431373],
    [ 0.70196078,  0.87058824,  0.41176471]]

datad = {}
datad['qualitative1'] = _qualitative1_data
datad['qualitative2'] = _qualitative2_data
