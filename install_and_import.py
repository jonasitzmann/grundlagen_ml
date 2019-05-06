import subprocess
dependencies ="""
numpy pandas matplotlib import-ipynb
""".split(' ')
for dependency in dependencies:
    cmd = 'pip3 install {}'.format(dependency)
    print(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import import_ipynb
from functools import partial