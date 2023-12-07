import os,sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import project_statics

Persian_DATA_BASE_URL = (
    "https://github.com/MobinZadkamali/Intent-Detection-and-Slot-Filling-for-Persian-Crosslingual-Training-for-Low-resource-Languages/"
    "tree/main/dataset/Persian_ATIS/"
)

English_DATA_BASE_URL = (
    "https://github.com/MobinZadkamali/Intent-Detection-and-Slot-Filling-for-Persian-Crosslingual-Training-for-Low-resource-Languages/"
    "tree/main/dataset/ATIS/"
)

from utils import parse_ourData_newformat

# raw file path, save destination path
parse_ourData_newformat(Persian_DATA_BASE_URL, English_DATA_BASE_URL, project_statics.SFID_pickle_files)