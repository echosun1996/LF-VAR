import json
import os
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import torch

try:
    from tap import Tap
except ImportError as e:
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<', file=sys.stderr, flush=True)
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<', file=sys.stderr, flush=True)
    time.sleep(5)
    raise e

import dist


class Args(Tap):
    dataset: str = None
    data_path: str = '/path/to/imagenet'
    fixed_csv_path: str = None
    tabsyn_csv_path: str = None
    exp_name: str = 'text'
    cross_infer: bool = False

    vfast: int = 0
    tfast: int = 0
    depth: int = 16
    ini: float = -1
    hd: float = 0.02
    aln: float = 0.5
    alng: float = 1e-5
    fp16: int = 0
    tblr: float = 1e-4
    tlr: float = None
    twd: float = 0.05
    twde: float = 0
    tclip: float = 2.
    ls: float = 0.0
    
    bs: int = 768
    batch_size: int = 0
    glb_batch_size: int = 0
    ac: int = 1
    
    ep: int = 250
    wp: float = 0
    wp0: float = 0.005
    wpe: float = 0.01
    sche: str = 'lin0'
    
    opt: str = 'adamw'
    afuse: bool = True
    
    saln: bool = False
    anorm: bool = True
    fuse: bool = True
    
    pn: str = '1_2_3_4_5_6_8_10_13_16'
    patch_size: int = 16
    patch_nums: tuple = None
    resos: tuple = None
    
    data_load_reso: int = None
    mid_reso: float = 1.125
    hflip: bool = False
    workers: int = 0
    
    pg: float = 0.0
    pg0: int = 4
    pgwp: float = 0
    
    cmd: str = ' '.join(sys.argv[1:])
    branch: str = subprocess.check_output(f'git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD', shell=True).decode('utf-8').strip() or '[unknown]'
    commit_id: str = subprocess.check_output(f'git rev-parse HEAD', shell=True).decode('utf-8').strip() or '[unknown]'
    commit_msg: str = (subprocess.check_output(f'git log -1', shell=True).decode('utf-8').strip().splitlines() or ['[unknown]'])[-1].strip()
    acc_mean: float = None
    acc_tail: float = None
    L_mean: float = None
    L_tail: float = None
    vacc_mean: float = None
    vacc_tail: float = None
    vL_mean: float = None
    vL_tail: float = None
    grad_norm: float = None
    cur_lr: float = None
    cur_wd: float = None
    cur_it: str = ''
    cur_ep: str = ''
    remain_time: str = ''
    finish_time: str = ''
    
    local_out_dir_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_output')
    tb_log_dir_path: str = '...tb-...'
    log_txt_path: str = '...'
    last_ckpt_path: str = '...'
    
    tf32: bool = True
    device: str = 'cpu'
    seed: int = None
    def seed_everything(self, benchmark: bool):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        if self.seed is None:
            torch.backends.cudnn.deterministic = False
        else:
            torch.backends.cudnn.deterministic = True
            seed = self.seed * dist.get_world_size() + dist.get_rank()
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
    same_seed_for_all_ranks: int = 0
    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed * dist.get_world_size() + dist.get_rank())
        return g
    
    local_debug: bool = 'KEVIN_LOCAL' in os.environ
    dbg_nan: bool = False
    
    def compile_model(self, m, fast):
        if fast == 0 or self.local_debug:
            return m
        return torch.compile(m, mode={
            1: 'reduce-overhead',
            2: 'max-autotune',
            3: 'default',
        }[fast]) if hasattr(torch, 'compile') else m
    
    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        for k in self.class_variables.keys():
            if k not in {'device'}:
                d[k] = getattr(self, k)
        return d
    
    def load_state_dict(self, d: Union[OrderedDict, dict, str]):
        if isinstance(d, str):
            d: dict = eval('\n'.join([l for l in d.splitlines() if '<bound' not in l and 'device(' not in l]))
        for k in d.keys():
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e
    
    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')
    
    def dump_log(self):
        if not dist.is_local_master():
            return
        if '1/' in self.cur_ep:
            with open(self.log_txt_path, 'w') as fp:
                json.dump({'is_master': dist.is_master(), 'name': self.exp_name, 'cmd': self.cmd, 'commit': self.commit_id, 'branch': self.branch, 'tb_log_dir_path': self.tb_log_dir_path}, fp, indent=0)
                fp.write('\n')
        
        log_dict = {}
        for k, v in {
            'it': self.cur_it, 'ep': self.cur_ep,
            'lr': self.cur_lr, 'wd': self.cur_wd, 'grad_norm': self.grad_norm,
            'L_mean': self.L_mean, 'L_tail': self.L_tail, 'acc_mean': self.acc_mean, 'acc_tail': self.acc_tail,
            'vL_mean': self.vL_mean, 'vL_tail': self.vL_tail, 'vacc_mean': self.vacc_mean, 'vacc_tail': self.vacc_tail,
            'remain_time': self.remain_time, 'finish_time': self.finish_time,
        }.items():
            if hasattr(v, 'item'): v = v.item()
            log_dict[k] = v
        with open(self.log_txt_path, 'a') as fp:
            fp.write(f'{log_dict}\n')
    
    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'


def init_dist_and_get_args():
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break
    args = Args(explicit_bool=True).parse_args(known_only=True)
    if args.local_debug:
        args.pn = '1_2_3'
        args.seed = 1
        args.aln = 1e-2
        args.alng = 1e-5
        args.saln = False
        args.afuse = False
        args.pg = 0.8
        args.pg0 = 1
    else:
        if args.data_path == '/path/to/imagenet':
            raise ValueError(f'{"*"*40}  please specify --data_path=/path/to/imagenet  {"*"*40}')
    
    if len(args.extra_args) > 0:
        print(f'======================================================================================')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================')
        print(f'======================================================================================\n\n')
    
    from utils import misc
    os.makedirs(args.local_out_dir_path, exist_ok=True)
    misc.init_distributed_mode(local_out_path=args.local_out_dir_path, timeout=30)
    
    args.set_tf32(args.tf32)
    args.seed_everything(benchmark=args.pg == 0)
    
    args.device = dist.get_device()
    if args.pn == '256':
        args.pn = '1_2_3_4_5_6_8_10_13_16'
    elif args.pn == '512':
        args.pn = '1_2_3_4_6_9_13_18_24_32'
    elif args.pn == '1024':
        args.pn = '1_2_3_4_5_7_9_12_16_21_27_36_48_64'
    args.patch_nums = tuple(map(int, args.pn.replace('-', '_').split('_')))
    args.resos = tuple(pn * args.patch_size for pn in args.patch_nums)
    args.data_load_reso = max(args.resos)
    
    bs_per_gpu = round(args.bs / args.ac / dist.get_world_size())
    args.batch_size = bs_per_gpu
    args.bs = args.glb_batch_size = args.batch_size * dist.get_world_size()
    args.workers = min(max(0, args.workers), args.batch_size)
    
    args.tlr = args.ac * args.tblr * args.glb_batch_size / 256
    args.twde = args.twde or args.twd
    
    if args.wp == 0:
        args.wp = args.ep * 1/50
    
    if args.pgwp == 0:
        args.pgwp = args.ep * 1/300
    if args.pg > 0:
        args.sche = f'lin{args.pg:g}'
    
    args.log_txt_path = os.path.join(args.local_out_dir_path, 'log.txt')
    args.last_ckpt_path = os.path.join(args.local_out_dir_path, f'ar-ckpt-last.pth')
    _reg_valid_name = re.compile(r'[^\w\-+,.]')
    tb_name = _reg_valid_name.sub(
        '_',
        f'tb-VARd{args.depth}'
        f'__pn{args.pn}'
        f'__b{args.bs}ep{args.ep}{args.opt[:4]}lr{args.tblr:g}wd{args.twd:g}'
    )
    args.tb_log_dir_path = os.path.join(args.local_out_dir_path, tb_name)
    
    return args