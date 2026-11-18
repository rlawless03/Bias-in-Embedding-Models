from weat import *

filename = "black__white_female_pleasantness.txt"
targets_1 = "targets/lower_black_american_female_names.txt"
targets_2 = "targets/lower_white_american_male_names.txt"
attributes_1 = "attributes/unpleasant.txt"
attributes_2 = "attributes/pleasant.txt"

generate_weat_results(filename, targets_1, targets_2, attributes_1, attributes_2)