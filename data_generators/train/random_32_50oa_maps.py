import numpy as np
import random
import os
import json
import multiprocessing as mp
import math
import gzip
import argparse
import sys
from pathlib import Path

THREADS = 20
OUT_DIM = 32
MAX_AGENTS = 50
NUM_WORLDS = 10000
MAX_COST_POSSIBLE = OUT_DIM * OUT_DIM

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

def generate_mapf_instances(worlds, idx, output_dir, od_mstar3_dir):
  sys.path.append(od_mstar3_dir)
  sys.path.append(str(Path(od_mstar3_dir).parent))
  import cpp_mstar
  
  print('Thread {} started.'.format(idx))
  for world_num in worlds:
    world = np.random.uniform(0, 1, (OUT_DIM, OUT_DIM))
    
    obs_den_random = random.uniform(0, 1)
    if obs_den_random <= 0.166:
      obs_den = 0.0
    elif obs_den_random <= 0.166*2:
      obs_den = 0.1
    elif obs_den_random <= 0.166*3:
      obs_den = 0.2
    elif obs_den_random <= 0.166*4:
      obs_den = 0.3
    elif obs_den_random <= 0.166*5:
      obs_den = 0.4
    else:
      obs_den = 0.5
    
    if world_num % 10 == 0:
      print('Working on World: {} Obstacle Density: {}'.format(world_num, obs_den))
    
    world[world <= obs_den] = -1
    world[world > obs_den] = 0
    world[world == -1] = 1
    obs = [(x, y) for x,y in zip(*np.where(world == 1))]
    available = [(x, y) for x,y in zip(*np.where(world == 0))]
          
    num_agents = 2
    while num_agents <= MAX_AGENTS:
      #print('Working on no. agents: ', num_agents)
      free = available.copy()
      
      init_pos = []
      goals = []
      all_costs_to_go = []
      
      while len(init_pos) < num_agents:
        random.shuffle(free)
        start = free[0]
        goal = free[1]
        
        try:
          cost_to_go = np.array(cpp_mstar.find_all_costs_to_go(world, [goal]))[0]
          if cost_to_go[start] > MAX_COST_POSSIBLE:
            raise Exception
        except:
          continue
        
        init_pos.append(start)
        goals.append(goal)
        all_costs_to_go.append(cost_to_go)
        
        free = free[2:]
        
      output = dict()
      output['world'] = world
      output['obs_den'] = obs_den
      output['obs'] = obs
      output['init_pos'] = init_pos
      output['goals'] = goals
      output['map_dim'] = (OUT_DIM, OUT_DIM)
      output['num_agents'] = num_agents
      output['all_costs_to_go'] = np.array(all_costs_to_go)
      
      out_file = output_dir + str(world_num) + '_' + str(int(obs_den*10)) + '_' + str(num_agents) + '.json.gz'
      with gzip.open(out_file, 'wt', encoding='ascii') as fout:
          json.dump(output, fout, cls=NumpyEncoder)
      
      num_agents += 1
      
  return idx

def log_result(idx):
    print('Completed:', idx)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-outputdir', required=True)
  parser.add_argument('-odmstardir', required=True)
  args = parser.parse_args()
  
  output_dir = args.outputdir
  od_mstar3_dir = args.odmstardir
  
  os.makedirs(output_dir, exist_ok=True)
  
  files = [i for i in range(NUM_WORLDS)]
  pool = mp.Pool(THREADS)
  results = []
  
  split_size = math.ceil(len(files)/THREADS)
  files_split = []
  idx = 0
  while 1:
    files_split.append(files[idx: idx + split_size])
    idx += split_size
    if idx > len(files):
      break

  for i in range(min(THREADS, len(files_split))):
    future = pool.apply_async(generate_mapf_instances, [files_split[i], i, output_dir, od_mstar3_dir], callback=log_result)
    results.append(future)
    
  for r in results:
    r.get()
