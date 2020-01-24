from __future__ import division
from os.path import expanduser
from random import choice, randint
import numpy as np
import sys


mydir = expanduser("~/GitHub/LearnMachineLearn/healthcareai")

outlist = ['patient_id', 'sex', 'age', 'diabetes', 'diabetes_type', 'weight',
           'overweight', 'yrs_smoker']

outstr = str(outlist).strip('[]')
outstr = outstr.replace(" ", "")
outstr = outstr.replace("'", "") 

OUT = open(mydir + '/SimData/SimData.csv', 'w+')
OUT.write(outstr+'\n')


for p_id in range(1000):
    
    sex = choice(['m', 'f'])
    sex_based_weight_bias = 1
    
    if sex == 'f':
        sex_based_weight_bias = 0.8
        
    age = randint(2, 85)
    # could be informed by population structure
    
    # expected weight according to fitted curve
    exp_weight = sex_based_weight_bias * (-0.07*(age-55)**2 + 220)
    # allow observed weight to vary normally around expected weight
    obs_weight = exp_weight + np.random.uniform(0, 1)*exp_weight
    
    # everyone starts diabetes free
    diabetes_type = 0
    
    # allow being type 1 (only) diabetic to be a low probability random event
    # with probability of 0.05 (5%)
    p = 0.05
    diabetes_type = np.random.binomial(1, p)
    
    if diabetes_type == 0:
        
        # allow being type 2 (only) diabetic to a consequence of being overweight
        overweight = obs_weight/exp_weight
        p = 1 / (1 + np.exp(-8*overweight+12))
        if np.random.binomial(1, p) == 1:
            diabetes_type = 2
    
            # allowing being type 1 & 2 (i.e., 3) diabetic to be a consequence of being
            # overweight and getting older
            p = 1 / (1 + np.exp(-0.1*overweight+4))
            if np.random.binomial(1, p) == 1:
                diabetes_type = 3
                print('yes')
    
    diabetes = 'N'
    if diabetes_type > 0:
        diabetes = 'Y'
        
    yrs_smoker = 0
    
    outlist = [p_id, sex, age, diabetes, diabetes_type, obs_weight, overweight,
               yrs_smoker]
        
    outstr = str(outlist).strip('[]')
    outstr = outstr.replace(" ", "") 
    outstr = outstr.replace("'", "") 
    OUT.write(outstr+'\n')

OUT.close()
    
