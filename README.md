# RACER: Data Generation Pipeline
Rich Language-guided Failure Recovery Data Augmentation Pipeline

<div style="text-align: center;">
  <img src="./data_pipeline_final-website.png" alt="Local Image" width="700" height="450">
</div>

## Getting Started

### Install RACER Data Gen

- **Step 1:**
```
conda create -name racer_data python=3.9
conda activate racer_data
```

- **Step 2:**
Follow Steps 2 to 4 from [RACER](https://github.com/sled-group/RACER).

- **Step 3:**
```
cd <PATH_TO_RACER_DATA_GEN>
git submodule update --init
pip install -e .
pip install -e libs/PyRep
pip install -e libs/RLbench 
pip install -e libs/YARR 
pip install -e libs/peract_colab
```

### Generate Data

Code coming soon~

For more information, please contact Jayjun Lee <jayjun@umich.edu>
