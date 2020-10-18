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
#                                                                                                 #
# Author: Abhijit Bendale (abendale@vast.uccs.edu)                                                #
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



import os
import glob
import time
import scipy as sp
from scipy.io import loadmat, savemat
import pathlib
import shutil

from OSDN.openmax_utils import getlabellist, get_train_labels
from config import Config

opt = Config()
featurefilepath = opt.feature_path


def compute_mean_vector(category_name, labellist, layer='feature'):
    featurefile_list = glob.glob('%s/%s/*.mat' % (featurefilepath, category_name))
    if not featurefile_list:
        return
    # gather all the training samples for which predicted category
    # was the category under consideration
    correct_features = []
    for featurefile in featurefile_list:
        try:
            img_arr = loadmat(featurefile)
            # extract only corrct label features
            predicted_category = labellist[img_arr['feature'].argmax()]
            if predicted_category == category_name:
                correct_features += [img_arr[layer]]

        except TypeError:
            continue
    # Now compute channel wise mean vector
    channel_mean_vec = []
    for channelid in range(1):
        channel = []
        for feature in correct_features:
            print(feature.shape)
            channel += [feature[channelid, :]]
        channel = sp.asarray(channel)
        assert len(correct_features) == channel.shape[0]
        # Gather mean over each channel, to get mean channel vector
        channel_mean_vec += [sp.mean(channel, axis=0)]

    # this vector contains mean computed over correct classifications
    # for each channel separately
    channel_mean_vec = sp.asarray(channel_mean_vec)
    print('%s.mat' % os.path.join(opt.mean_files_path, category_name))
    output_dir = pathlib.Path(opt.mean_files_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    savemat('%s.mat' % str(output_dir.joinpath(category_name)), {'{}'.format(category_name): channel_mean_vec})


def main():
    if pathlib.Path(opt.mean_files_path).exists():
        shutil.rmtree(opt.mean_files_path)
    st = time.time()
    # labellist = getlabellist(opt.criteria_list)
    labellist = get_train_labels(opt.train_root, opt.criteria_list)
    for category_name in labellist:
        compute_mean_vector(category_name, labellist)
    print("Total time %s secs" %(time.time() - st))


if __name__ == "__main__":
    main()
