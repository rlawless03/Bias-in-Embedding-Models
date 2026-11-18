from weat import *

filename = "black_white_male_competence.txt"
targets_1 = "targets/lower_black_american_female_names.txt"
targets_2 = "targets/lower_white_american_male_names.txt"
attributes_1 = "attributes/incompetence.txt"
attributes_2 = "attributes/competence.txt"

generate_weat_results(filename, targets_1, targets_2, attributes_1, attributes_2)