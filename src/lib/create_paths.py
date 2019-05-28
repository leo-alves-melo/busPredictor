# Importing packages
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import pprint
import json
import pandas as pd
import numpy as np
#import arrow
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
#from functools import reduce
from math import sin, cos, sqrt, atan2, radians
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm
from datetime import timedelta
import copy
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from constants import *
from data_collector import *

df = pd.read_csv('data/processed_data.csv')
paths_df = create_paths(df)
paths_df.to_csv('data/paths.csv', index=False)