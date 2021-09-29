# Subdimensional Expansion Using Attention-Based Learning For Multi-Agent Path Finding
**Multi-Agent Path Finding (MAPF)** finds conflict-free
paths for multiple agents from their respective start to
goal locations. MAPF is challenging as the joint configuration
space grows exponentially with respect to the number of agents.
Among **MAPF** planners, search-based methods, such as **CBS**
and **M***, effectively bypass the curse of dimensionality by
employing a **dynamically-coupled** strategy: agents are planned
in a fully decoupled manner at first, where potential conflicts
between agents are ignored; and then agents either follow their
individual plans or are coupled together for planning to resolve
the conflicts between them. In general, the number of conflicts
to be resolved decides the run time of these planners and most
of the existing work focuses on how to efficiently resolve these
conflicts. In this work, we take a different view and **aim to
reduce the number of conflicts (and thus improve the overall
search efficiency) by improving each agent’s individual plan**. By
leveraging a **Visual Transformer**, we develop a **learning-based
single-agent planner**, which plans for a single agent while paying
attention to both the structure of the map and other agents
with whom conflicts may happen. We then develop a novel
multi-agent planner called **LM*** by integrating this learning-based
single-agent planner with M*. Our results show that
for both **“seen”** and **“unseen”** maps, in comparison with M*,
LM* has fewer conflicts to be resolved and thus, **runs faster**
and enjoys **higher success rates**. We empirically show that
MAPF solutions computed by LM* are **near-optimal**.

<p align="center">
<img src="https://github.com/wonderren/pymahl/blob/cf1d10eff2f6e0d158163ac406fc1926d145098e/lmstar.png" width="50%">
</p>
  
The above figure is an illustration of Learning-Assisted M* (LM*). At
every time step, each agent shares its observations with the
attention-based model and the model predicts the action for
each agent individually by attending to the structure of the
map and other agents’ information. The agents follow the
predicted actions if there are no conflicts. Otherwise the
agents in conflict are coupled together by planning in their
joint configuration space, just like M*.

For details see [Subdimensional Expansion Using Attention-Based Learning For Multi-Agent Path Finding](https://arxiv.org/) by Lakshay Virmani, Zhongqiang Ren, Sivakumar Rathinam and Howie Choset.

## Key files
- `data_generators/`
  - `train/`: 
    - `random_32_50oa_maps.py` - Generate MAPF instances containing upto 50 agents with probability of each cell being marked as an obstacle being randomly chosen from 0%-50%.
    - `random_32_50oa_solver.py` - Solve MAPF instances generated using ODrM* with an inflation of 1.1.
    - `random_32_50oa_batch.py` - Extract single-agent inputs from solved MAPF instances and group them into batches each containing 16 such inputs.
  - `test/`: 
    - `seen_maps.py` - Generate MAPF instances using 100 randomly selected maps from the train set.
    - `mstar_maps_unseen_same_type.py` - Generate MAPF instances as per the M* paper with probability of each cell being marked as an obstacle set to 20%.
    - `mapf_benchmark_room_maze.py` - Generate MAPF instances using room and maze maps from the MAPF Benchmark dataset.
- `models/`:
  - `trainer.py` - Setup and train an attention-based model using the generated training data.
- `lmstar/`:
  - `test_lmstar.py` - Use a pretrained model to run subdimensional expansion using attention-based learning for multi-agent path finding. (LM*)

## Setting up data generators
- To run the data generators we first need to compile the C++ implementation of [ODrM*](https://github.com/gswagner/mstar_public). This can be done by executing the following command in the `data_generators/od_mstar3/` directory:
```
python3 setup.py build_ext --inplace
```
- To check run python3 in the `data_generators` directory and execute the following commands:
```
import sys
sys.path.append('./od_mstar3/')
import cpp_mstar
```

## Generating training data
- Training data can be generated using the following three commands in the `data_generators/train/` directory. We first generate the mapf instances which are then solved using ODrM* with 1.1 inflation. We then extract single-agent inputs and create batches each containing 16 such inputs for the model.
```
python3 random_32_50oa_maps.py -outputdir <output dir> -odmstardir <dir containing compiled odmstar>
python3 random_32_50oa_solver.py -inputdir <dir containing generated mapf instances> -outputdir <output dir> -odmstardir <dir containing compiled odmstar>
python3 random_32_50oa_batch.py -inputdir <dir containing solved mapf instances> -outputdir <output dir>
```

## Training the model
- The model can be trained using the following command in the `models/` directory:
```
python3 trainer.py -inputdir <dir containing batches of single-agent inputs> -device <gpu for training e.g. cuda:0> -epochs <epochs to train> -modelpath <path for saving trained model>
```

## Testing LM*
- LM* can be used for solving MAPF instances using the following commands in the `lmstar/` directory:
```
python3 test_lmstar.py -inputdir <dir containing mapf test instances> -outputdir <output dir> -modeldir <dir containing model.py file> -trainedmodel <path to trained model> -inflation <inflation rate>
```
