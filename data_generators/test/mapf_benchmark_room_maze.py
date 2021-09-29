import numpy as np
import json
import os
import sys
import argparse

INPUT_MAPS_DIR = './benchmark_maps/'

def create_map_to_scenario_dict():
  input_maps = [INPUT_MAPS_DIR + x for x in os.listdir(INPUT_MAPS_DIR) if '.map' in x]
  map_to_scen = {}
  for input_map in input_maps:
    map_name = input_map.split('/')[-1].split('.')[0]
    even_dir = INPUT_MAPS_DIR + map_name + '-scen-even/'
    even_scen = [even_dir + x for x in os.listdir(even_dir)]
    rand_dir = INPUT_MAPS_DIR + map_name + '-scen-random/'
    rand_scen = [rand_dir + x for x in os.listdir(rand_dir)]
    map_to_scen[input_map] = {}
    map_to_scen[input_map]['even'] = even_scen
    map_to_scen[input_map]['random'] = rand_scen
  return map_to_scen

def load_map(map_file):
  """
  load map from given file
  """
  grids = np.zeros((2,2))
  with open(map_file,'r') as f:
    lines = f.readlines()
    lidx = 0
    nx = 0
    ny = 0
    for line in lines:
      if lidx == 1:
        a = line.split(" ")
        nx = int(a[1])
      if lidx == 2:
        a = line.split(" ")
        ny = int(a[1])
      if lidx == 4:
        grids = np.zeros((nx,ny))
      if lidx >= 4: # map data begin
        x = lidx - 4
        y = 0
        a = line.split("\n")
        for ia in str(a[0]):
          if ia == "." or ia == "G":
            grids[x,y] = 0
          else:
            grids[x,y] = 1
          y = y+1
      lidx = lidx + 1
  return grids

def load_scenario(map_file, scen_file, cvecs=[], cgrids=[]):
  """
  load map and scen from given files and return a test_case dict.
  """
  grids = load_map(map_file)
  with open(scen_file,'r') as f:
    lines = f.readlines()
    test_case_dict = dict() 
    sx = list()
    sy = list()
    gx = list()
    gy = list()
    d_list = list()
    for line in lines:
      a = line.split("\t")
      if len(a) != 9: # an invalid line, skip it
        continue
      else: # same as last case
        sx.append(int(a[4]))
        sy.append(int(a[5]))
        gx.append(int(a[6]))
        gy.append(int(a[7]))
        d_list.append(1) # assume homo case
  test_case_dict = dict()
  test_case_dict["sx"] = sx
  test_case_dict["sy"] = sy
  test_case_dict["gx"] = gx
  test_case_dict["gy"] = gy
  test_case_dict["d_list"] = d_list
  test_case_dict["grids"] = grids
  test_case_dict["cost_grids"] = cgrids
  test_case_dict["cost_vecs"] = cvecs
  return test_case_dict

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)

def generate_mapf_instances(map_to_scen, output_dir, od_mstar3_dir):
  sys.path.append(od_mstar3_dir)
  import cpp_mstar
  
  for map in map_to_scen:
    scens = map_to_scen[map]
    for scen_type in scens:
      for scen_idx, scen in enumerate(scens[scen_type]):
        test_case = load_scenario(map, scen)
        map_name = map.split('/')[-1].split('.')[0]
        world = test_case['grids']
        
        obs = []
        for i in range(32):
          for j in range(32):
            if world[i][j] == 1:
              obs.append((i, j))
              
        init_pos = []
        goals = []
        for i in range(len(test_case['sx'])):
          start = (test_case['sx'][i], test_case['sy'][i])
          goal = (test_case['gx'][i], test_case['gy'][i])
          if start in obs or goal in obs:
            continue
          init_pos.append(start)
          goals.append(goal)
        
        all_costs_to_go = np.array(cpp_mstar.find_all_costs_to_go(world, goals))
        
        out = {}
        for i in range(5, len(init_pos), 5):
          out['map_dim'] = (32,32)
          out['world'] = world
          out['init_pos'] = init_pos[:i]
          out['goals'] = goals[:i]
          out['obs'] = obs
          out['map_name'] = map_name
          out['scen'] = scen_type
          out['all_costs_to_go'] = all_costs_to_go[:i]
          
          out_name = map_name + '_' + scen_type + '_' + str(scen_idx) + '_' + str(i) + '.json'
          with open(output_dir + out_name, 'w') as fout:
            json.dump(out, fout, cls=NumpyEncoder)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-outputdir', required=True)
  parser.add_argument('-odmstardir', required=True)
  args = parser.parse_args()
  
  output_dir = args.outputdir
  od_mstar3_dir = args.odmstardir
  
  os.makedirs(output_dir, exist_ok=True)
  
  map_to_scen = create_map_to_scenario_dict()
  generate_mapf_instances(map_to_scen, output_dir, od_mstar3_dir)

