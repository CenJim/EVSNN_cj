#!/bin/bash
source ~/.bashrc
conda activate snnrec 
python rec_snn.py -network EVSNN_LIF_final -path_to_pretrain_models ./pretrained_models/EVSNN.pth

