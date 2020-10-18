# -*- coding: utf-8 -*-

###################################################################################################
# Copyright (c) 2016 , Regents of the University of Colorado on behalf of the University          #
# of Colorado Colorado Springs.  All rights reserved.                                             #
#                                                                                                 #
# Redistribution and use in source and binary forms, with or without modification,                #
# are permitted provided that the following conditions are met:                                   #
#                                                                                                 #
# 1. Redistributions of source code must retain the above copyright notice, this                  #
# list of conditions and the following disclaimer.                                                #
#                                                                                                 #
# 2. Redistributions in binary form must reproduce the above copyright notice, this list          #
# of conditions and the following disclaimer in the documentation and/or other materials          #
# provided with the distribution.                                                                 #
#                                                                                                 #
# 3. Neither the name of the copyright holder nor the names of its contributors may be            #
# used to endorse or promote products derived from this software without specific prior           #
# written permission.                                                                             #
#                                                                                                 #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY             #
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF         #
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          #
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,            #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF     #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,           #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS           #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    #
#                                                                                                 #
# Author: Abhijit Bendale (abendale@vast.uccs.edu)                                                #
#                                                                                                 #
# If you use this code, please cite the following works                                           #
#                                                                                                 #
# A. Bendale, T. Boult “Towards Open Set Deep Networks” IEEE Conference on                        #
# Computer Vision and Pattern Recognition (CVPR), 2016                                            #
#                                                                                                 #
# Notice Related to using LibMR.                                                                  #
#                                                                                                 #
# If you use Meta-Recognition Library (LibMR), please note that there is a                        #
# difference license structure for it. The citation for using Meta-Recongition                    #
# library (LibMR) is as follows:                                                                  #
#                                                                                                 #
# Meta-Recognition: The Theory and Practice of Recognition Score Analysis                         #
# Walter J. Scheirer, Anderson Rocha, Ross J. Micheals, and Terrance E. Boult                     #
# IEEE T.PAMI, V. 33, Issue 8, August 2011, pages 1689 - 1695                                     #
#                                                                                                 #
# Meta recognition library is provided with this code for ease of use. However, the actual        #
# link to download latest version of LibMR code is: http://www.metarecognition.com/libmr-license/ #
###################################################################################################

from scipy.io import loadmat

from OSDN.openmax_utils import *

try:
    import libmr
except ImportError:
    print("LibMR not installed or libmr.so not found")
    print("Install libmr: cd libMR/; ./compile.sh")
    sys.exit()


def weibull_tailfitting(meanfiles_path, distancefiles_path, labellist,
                        tailsize=20,
                        distance_type='eucos'):
                        
    """ Read through distance files, mean vector and fit weibull model for each category

    Input:
    --------------------------------
    meanfiles_path : contains path to files with pre-computed mean-activation vector
    distancefiles_path : contains path to files with pre-computed distances for images from MAV
    labellist : criterila_list

    Output:
    --------------------------------
    weibull_model : Perform EVT based analysis using tails of distances and save
                    weibull model parameters for re-adjusting softmax scores    
    """
    
    weibull_model = {}
    # for each category, read meanfile, distance file, and perform weibull fitting
    for category in labellist:
        weibull_model[category] = {}
        distance_mat_path = pathlib.Path(distancefiles_path).joinpath('{}_distance.mat'.format(category))
        mean_train_vec_path = pathlib.Path(meanfiles_path).joinpath('{}.mat'.format(category))
        if not (distance_mat_path.exists() and mean_train_vec_path.exists()):
            continue
        distance_scores = loadmat(distance_mat_path)[distance_type]
        meantrain_vec = loadmat(mean_train_vec_path)
        weibull_model[category]['distances_%s' % distance_type] = distance_scores
        weibull_model[category]['mean_vec'] = meantrain_vec
        weibull_model[category]['weibull_model'] = []
        for channel in range(1):
            mr = libmr.MR()
            tailtofit = sorted(distance_scores[channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category]['weibull_model'] += [mr]
    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    """ Query through dictionary for Weibull model.
    Return in the order: [mean_vec, distances, weibull_model]
    
    Input:
    ------------------------------
    category_name : name of ImageNet category in WNET format. E.g. n01440764
    weibull_model: dictonary of weibull models for 
    """
    category_weibull = [[], [], []]
    try:
        category_weibull += [weibull_model[category_name]['mean_vec'][category_name]]
        category_weibull += [weibull_model[category_name]['distances_%s' % distance_type]]
        category_weibull += [weibull_model[category_name]['weibull_model']]
    except KeyError:
        pass

    return [weibull_model[category_name]['mean_vec'],
            weibull_model[category_name]['distances_{}'.format(distance_type)],
            weibull_model[category_name]['weibull_model']]
