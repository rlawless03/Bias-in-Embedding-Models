from weat import *

filename = "p_black_white_competence.txt"
targets_1 = "targets/combined_black_american_names.txt"
targets_2 = "targets/combined_white_american_names.txt"
attributes_1 = "attributes/incompetence.txt"
attributes_2 = "attributes/competence.txt"

generate_weat_results(filename, targets_1, targets_2, attributes_1, attributes_2)