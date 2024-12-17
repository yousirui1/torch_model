import os

current_dir = os.path.dirname(os.path.realpath(__file__))
#parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
os.sys.path.append(current_dir)
print(current_dir)

from model import *


