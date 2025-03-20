#!/bin/bash
 
#python3 falsifier.py -p ./scenarios-dddas/intermittent_attack_mab.scenic -r outputs/intermittent_mab -s mab
#python3 falsifier.py -p ./scenarios-dddas/intermittent_attack_ce.scenic -r outputs/intermittent_ce -s ce
#python3 falsifier.py -p ./scenarios-dddas/persistent_attack_mab.scenic -r outputs/persistent_mab -s mab
#python3 falsifier.py -p ./scenarios-dddas/persistent_attack_ce.scenic -r outputs/persistent_ce -s ce

python3 mult_obj_falsifier.py -p ./scenarios-dddas/intermittent_attack_mab.scenic -s mab -r outputs/intermittent_mab_w_priority_1
python3 mult_obj_falsifier.py -p ./scenarios-dddas/persistent_attack_mab.scenic -s mab -r outputs/persistent_mab_w_priority_1
