# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 10:49
# @Author  : Naynix
# @File    : parallel.py
from datetime import datetime
from functools import partial
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor


def partition(lst, n):
    length = len(lst)
    size = length // n
    res = []
    for i in range(n-1):
        res.append(lst[i*size:size*(i+1)])
    res.append(lst[size*(n-1):])
    return res


def multi_process(data, nthread, func, use_thread=False, res_type='list', **kwargs):
    if len(data) < nthread*10:
        if kwargs:
            res = func(data, kwargs=kwargs)
        else:
            res = func(data)
        return res

    t_start = datetime.now()
    thread_data = partition(data, nthread)
    if use_thread:
        procs = ThreadPoolExecutor(nthread)
    else:
        procs = ProcessPoolExecutor(nthread)

    if kwargs:
        f_func = partial(func, kwargs=kwargs)
    else:
        f_func = partial(func)
    thread_res = procs.map(f_func, thread_data)
    procs.shutdown(wait=True)

    res = None
    if res_type == 'list':
        res = []
        for item in thread_res:
            res.extend(item)
    elif res_type == 'set':
        res = set()
        for item in thread_res:
            res = res | item
    elif res_type == 'dict':
        res = dict()
        for item in thread_res:
            for key in item:
                if isinstance(item[key], set):
                    res.setdefault(key, set()).update(item[key])
                elif isinstance(item[key], list):
                    res.setdefault(key, list()).extend(item[key])
                else:
                    res[key] = item[key]
    else:
        res = None
    time = datetime.now() - t_start
    print('Time to preprocess the data : {} min'.format(time.seconds/60))
    return res