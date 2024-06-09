#!/bin/bash
source ~/.bashrc
conda activate snnrec 
python rec_snn.py -network EVSNN_LIF_final -path_to_pretrain_models ./pretrained_models/EVSNN.pth \
    -path_to_event_files /home/s2491540/dataset/IJRR/poster_6dof_txt/events.txt \
    -height 180 -width 240

