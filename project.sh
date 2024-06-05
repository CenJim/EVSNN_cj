#!/bin/bash
source ~/.bashrc
conda activate snnrec 
python rec_snn.py -network EVSNN_LIF_final -path_to_pretrain_models ./pretrained_models/EVSNN.pth \
    -path_to_event_files /home/s2491540/dataset/DSEC/test/thun_01_a/thun_01_a_events_left/events.h5

