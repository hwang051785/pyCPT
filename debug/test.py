import sys
sys.path.append("~/PycharmProjects/pyMRF")

import pre_process

data_path = "../test_data/Schop_H35.tif"

(data, coord) = pre_process.pre_process(data_path)

print(coord)



