"""Run a notebook's code cells up to and including its training cell, in this folder (a parallel fold worker).
Usage: ONLY_RUNS=0,4 CUDA_VISIBLE_DEVICES=0 python run_cells.py <notebook.ipynb>"""
import json, sys, matplotlib
matplotlib.use('Agg')   # no display on a server
cells = [''.join(c['source']) for c in json.load(open(sys.argv[1], encoding='utf-8'))['cells'] if c['cell_type'] == 'code']
last = next(i for i, s in enumerate(cells) if "ONLY = os.environ.get('ONLY_RUNS')" in s)   # the training cell
ns = {'__name__': '__main__'}
for i, s in enumerate(cells[:last + 1]):
    print(f'===== code cell {i}', flush=True)
    exec(compile(s, f'cell {i}', 'exec'), ns)
