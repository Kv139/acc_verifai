#!/bin/bash

python3 falsifier.py -p ./scenarios-dddas/persistent_attack.scenic -r outputs/persistent_ce -s ce 
python3 falsifier.py -p ./scenarios-dddas/intermittent_attack.scenic -r outputs/intermittent_ce -s ce

