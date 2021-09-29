import numpy as np
import os
import json
import traceback
import multiprocessing as mp
import math
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import gzip
import sys
import argparse
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)

def solve_mapf_instances(files, idx, output_dir, od_mstar3_dir):
  sys.path.append(od_mstar3_dir)
  sys.path.append(str(Path(od_mstar3_dir).parent))
  import cpp_mstar
  
  track_failed = dict()
  for file_idx, file in enumerate(files):
    if file_idx % 10 == 0:
      print('Working on file: {}, for thread: {}'.format(file_idx, idx))
    
    try:
      with gzip.open(file, 'rt', encoding='ascii') as fin:
        data = json.load(fin)  
    except Exception as e:
      print(file)
      print(e)
      continue
    
    world = np.array(data['world'])
    num_agents = data['num_agents']
    init_pos = data['init_pos']
    goals = data['goals']
    world_num = file.split('/')[-1].split('_')[0]
    
    if world_num in track_failed:
      if num_agents > track_failed[world_num]:
        print('Skipping.')
        continue
    
    try:
      path = cpp_mstar.find_path(world, init_pos, goals, inflation=1.1, time_limit=300)
    except Exception:
      traceback.print_exc()
      track_failed[world_num] = min(num_agents, track_failed.get(world_num, 100))
      continue
    
    data['path'] = path
    
    out_file = output_dir + file.split('/')[-1]
    with gzip.open(out_file, 'wt', encoding='ascii') as fout:
        json.dump(data, fout, cls=NumpyEncoder)

def log_result(idx):
    print('Completed:', idx)
            
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-inputdir', required=True)
  parser.add_argument('-outputdir', required=True)
  parser.add_argument('-odmstardir', required=True)
  args = parser.parse_args()
  
  input_dir = args.inputdir
  output_dir = args.outputdir
  od_mstar3_dir = args.odmstardir
  
  os.makedirs(output_dir, exist_ok=True)
  
  inputs = [input_dir + x for x in os.listdir(input_dir)]
  print('Total Inputs: ', len(inputs))
  inputs = sorted(inputs)
  
  threads = 20
  pool = mp.Pool(threads)
  results = []
  files = inputs

  split_size = math.ceil(len(files)/threads)
  files_split = []
  idx = 0
  while 1:
    files_split.append(files[idx: idx + split_size])
    idx += split_size
    if idx > len(files):
      break

  for i in range(min(threads, len(files_split))):
    future = pool.apply_async(solve_mapf_instances, [files_split[i], i, output_dir, od_mstar3_dir], callback=log_result)
    results.append(future)

  for r in results:
    r.get()

