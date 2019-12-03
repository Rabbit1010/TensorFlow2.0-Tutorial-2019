# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:11:34 2019

@author: Wei-Hsiang, Shen
"""

import sys

def Get_Object_Memory_Size(obj, seen=None):
    """Recursively finds size of objects"""

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([Get_Object_Memory_Size(v, seen) for v in obj.values()])
        size += sum([Get_Object_Memory_Size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += Get_Object_Memory_Size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([Get_Object_Memory_Size(i, seen) for i in obj])

    return size