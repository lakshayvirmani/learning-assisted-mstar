import numpy as np
import random
import os
import json
import sys
import argparse

OUT_DIM = 32
OBS_DEN = 0.2
MAX_AGENTS = 200
NUM_WORLDS = 100
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

def generate_mapf_instances(output_dir, od_mstar3_dir):
  sys.path.append(od_mstar3_dir)
  import cpp_mstar
  
  for world_num in range(NUM_WORLDS):
    print('Working on World: ', world_num)
    world = np.random.uniform(0, 1, (OUT_DIM, OUT_DIM))
    world[world <= OBS_DEN] = -1
    world[world > OBS_DEN] = 0
    world[world == -1] = 1
    obs = [(x, y) for x,y in zip(*np.where(world == 1))]
    available = [(x, y) for x,y in zip(*np.where(world == 0))]
          
    num_agents = 2
    while num_agents <= MAX_AGENTS:
      print('Working on no. agents: ', num_agents)
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
      output['obs'] = obs
      output['init_pos'] = init_pos
      output['goals'] = goals
      output['map_dim'] = (OUT_DIM, OUT_DIM)
      output['num_agents'] = num_agents
      output['all_costs_to_go'] = np.array(all_costs_to_go)
      
      out_file = output_dir + str(world_num) + '_' + str(num_agents) + '.json'
      with open(out_file, 'w') as fout:
          json.dump(output, fout, cls=NumpyEncoder)
      
      num_agents += 5

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-outputdir', required=True)
  parser.add_argument('-odmstardir', required=True)
  args = parser.parse_args()
  
  output_dir = args.outputdir
  od_mstar3_dir = args.odmstardir
  
  os.makedirs(output_dir, exist_ok=True)
  
  generate_mapf_instances(output_dir, od_mstar3_dir)