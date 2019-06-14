# -*- coding: utf-8 -*-

'''
等待gpu资源后，自动运行代码
from gpu import *
with wait_gpu():
    tensorflow代码
'''


import os, time
import datetime
import tensorflow as tf

__all__ = ['wait_gpu']


cnt = 0

def print_log(free_mem_rate, used_gpu_rate):
    
    global cnt
    if cnt%100 == 0:
        cnt = 0
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(now_time, "free_mem and used_gpu are: ", free_mem_rate, used_gpu_rate)
    cnt += 1

def get_gpu_state(gpu_param):
    free_mem_rate = gpu_param['memory.free']/gpu_param['memory.total']
    used_gpu_rate = float(gpu_param['utilization.gpu'].replace('%','').strip())/100.0
    print_log(free_mem_rate, used_gpu_rate)
    if free_mem_rate>0.8 and used_gpu_rate<0.12:
        return True
    else:
        return False

def parse(line,qargs):
    '''
    line:
        a line of text
    qargs:
        query arguments
    return:
        a dict of gpu infos
    Pasing a line of csv format text returned by nvidia-smi
    解析一行nvidia-smi返回的csv格式文本
    '''
    numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']#可计数的参数
    power_manage_enable=lambda v:(not 'Not Support' in v)#lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
    to_numberic=lambda v:float(v.upper().strip().replace('MIB','').replace('W',''))#带单位字符串去掉单位
    process = lambda k,v:((int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
    return {k:process(k,v) for k,v in zip(qargs,line.strip().split(','))}

def query_gpu(qargs=[]):
    '''
    qargs:
        query arguments
    return:
        a list of dict
    Querying GPUs infos
    查询GPU信息
    '''
    qargs =['index','gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit', 'utilization.gpu']+ qargs
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()
    return [parse(line,qargs) for line in results]

def by_power(d):
    '''
    helper function fo sorting gpus by power
    '''
    power_infos=(d['power.draw'],d['power.limit'])
    if any(v==1 for v in power_infos):
        # print('Power management unable for GPU {}'.format(d['index']))
        return 1
    return float(d['power.draw'])/d['power.limit']

class GPUManager():
    '''
    qargs:
        query arguments
    A manager which can list all available GPU devices
    and sort them and choice the most free one.Unspecified 
    ones pref.
    GPU设备管理器，考虑列举出所有可用GPU设备，并加以排序，自动选出
    最空闲的设备。在一个GPUManager对象内会记录每个GPU是否已被指定，
    优先选择未指定的GPU。
    '''
    def __init__(self,qargs=[]):
        '''
        '''
        self.qargs=qargs
        self.gpus=query_gpu(qargs)
        for gpu in self.gpus:
            gpu['specified']=False
        self.gpu_num=len(self.gpus)

    def _sort_by_memory(self,gpus,by_size=False):
        if by_size:
            # print('Sorted by free memory size')
            return sorted(gpus,key=lambda d:d['memory.free'],reverse=True)
        else:
            # print('Sorted by free memory rate')
            return sorted(gpus,key=lambda d:float(d['memory.free'])/ d['memory.total'],reverse=True)

    def _sort_by_power(self,gpus):
        return sorted(gpus,key=by_power)
    
    def _sort_by_custom(self,gpus,key,reverse=False,qargs=[]):
        if isinstance(key,str) and (key in qargs):
            return sorted(gpus,key=lambda d:d[key],reverse=reverse)
        if isinstance(key,type(lambda a:a)):
            return sorted(gpus,key=key,reverse=reverse)
        raise ValueError("The argument 'key' must be a function or a key in query args,please read the documention of nvidia-smi")

    def auto_choice(self,mode=0):
        '''
        mode:
            0:(default)sorted by free memory size
        return:
            a TF device object
        Auto choice the freest GPU device,not specified
        ones 
        自动选择最空闲GPU
        '''
        not_avaliable = True
        chosen_gpu, index = None, None
        while not_avaliable:
            for old_infos,new_infos in zip(self.gpus,query_gpu(self.qargs)):
                old_infos.update(new_infos)
            unspecified_gpus=[gpu for gpu in self.gpus if not gpu['specified']] or self.gpus
            if mode==0:
                # print('Choosing the GPU device has largest free memory...')
                chosen_gpu=self._sort_by_memory(unspecified_gpus,True)[0]
            elif mode==1:
                # print('Choosing the GPU device has highest free memory rate...')
                chosen_gpu=self._sort_by_power(unspecified_gpus)[0]
            elif mode==2:
                # print('Choosing the GPU device by power...')
                chosen_gpu=self._sort_by_power(unspecified_gpus)[0]
            else:
                # print('Given an unaviliable mode,will be chosen by memory')
                chosen_gpu=self._sort_by_memory(unspecified_gpus)[0]
            chosen_gpu['specified']=True
            index=chosen_gpu['index']
            # print(chosen_gpu)
            not_avaliable = not get_gpu_state(chosen_gpu)
            if not_avaliable:   time.sleep(6)
            
        print('Using GPU {i}:\n{info}'.format(i=index,info='\n'.join([str(k)+':'+str(v) for k,v in chosen_gpu.items()])))
        return tf.device('/gpu:{}'.format(index))

def wait_gpu():
    return GPUManager().auto_choice()
