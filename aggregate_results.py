#!/usr/bin/env python
# 
# All modification made by Walter Riviera
# 
# Author: Walter Riviera
# email: walteriviera@gmail.com
# role: AI-TSS lead for EMEA @Intel corporation

import os
import sys
import argparse
import math
import csv
import json

from optparse import OptionParser



def get_hyperparams(experiment_name=""):
    '''Given an experiment name, it returns the dictionary containin couples <param: value(s)>. Params and values are automatically inferred from the name
    experiment_name must follows the pattern': ppn_1-cps_24-intra_24-inter_2-bt_60-img_6982-lr_0.01-mom_0.9-bs_64-dpout_0.5_0.5_0.5_0.5_0.5_0.5_0.5_0.5 '''
    
    parameters=experiment_name.split("-")
    out={}
    for p in parameters:
        values = p.split("_")
        param = values[0]
        out[param]= values[1:]
        
    return out
    
    
def val2line(info={},settings={},mode="line"):
    
    '''take as input 2 dicts: info, settings.
    returns a list of values organized as follow:
    if mode=="header" --> all_keys in "settings" + all keys in "info".
    if mode=="line -->  all_values in "settings" + all values in "info" '''
    
    ## TODO: check param and return if empty
    
    if (mode=="header"):
        return ['processes per socket', 'cores per process','total processes',
                'intra thread', 'inter thread', 'blocktime', 'training images', 
                'learning rate', 'momentum', 'batch size', 'dropout', 
                'total epochs', 'steps per epoch', 'total training time', 
                'early stop iteration', 'val_loss', 'val_acc', 'val_dsc']
    
    else:
        line = [int(settings['ppn'][0]),int(settings['cps'][0]),int(info['total_job_ranks']),
               int(settings['intra'][0]),int(settings['inter'][0]),int(settings['bt'][0]),int(settings['img'][0]),
               float(settings['lr'][0]),float(settings['mom'][0]),int(settings['bs'][0]),settings['dpout'],
               int(info['total_epochs']),int(info['steps_x_epoch']),float(info['total_training_time']),
               int(info['early_stop_iteration']),float(info['val_loss']),float(info['val_acc']),float(info['val_dsc'])]     
     
        return line
        


def saveToSummary(outfname="",summary={},savepath=""):
    
    outputfile = os.path.join(savepath,outfname+".csv")
    
    # Sort results according to val_dsc/loss ratio (high dsc and low loss are best)
    ordered_summary = sorted(summary.items(), key=lambda x: x[1][2], reverse=True) 
    
    fieldnames=val2line({},{},"header")
    with open(outputfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",",quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(fieldnames)# write columns names
        
        for item in ordered_summary:
            training_info     = item[1][0]
            training_settings = item[1][1]      # equivalent of =ordered_summary[item][1] if dict wasn't sorted
            data = val2line(training_info, training_settings,"line")
        
            writer.writerow(data)



def load_jsonData(jsonfile_name="", filepath=""):
    
    ''' Load results and training_info dictionaries from filename defined as "finput" with extension ".json".'''

    inputfile = os.path.join(filepath,jsonfile_name)
    
    with open(inputfile, 'r') as f:
        training_info = json.load(f)
    
    #results = training_info["results"]
    training_info.pop("results", None)
            
    return training_info




def main():
    
    """Run the result aggregator"""

    usage = "Usage: aggregate_results.py [options]"
    parser = OptionParser(usage=usage)
    
    # ways to define the Input Path
    parser.add_option("-i","--input_path"  , dest="inputpath", default="", help="Input path")
    
    # ways to define the Output Path
    parser.add_option("-o","--output_path" , dest="savepath",default="",help="Save path for output files")
    
    # ways to define the Output Path
    parser.add_option("-f","--filename", dest="outputname",default="summary",help="Name of the output files")
    
    
    ## Parameter check
    opts, args = parser.parse_args()

    if len(args):
        parser.error("this script does not take any arguments")

    if not(opts.inputpath): 
        parser.error("please specify '-i'/'--input'")
        sys.exit()
    #-------------------------------------------------------------------------------------------------
    
    
    # Init I/O variables
    results_path    = opts.inputpath
    outputfile_path = opts.savepath
    outputfile_name = opts.outputname
    
    # Step_0: Get list of dir names within results_path
    experiments = [ f.name for f in os.scandir(results_path) if f.is_dir() ]

    # Step_1: Retrieve training information and set-up for each experiment
    output={}
    for n,exp in enumerate(experiments):

        if os.path.isfile(os.path.join(results_path,exp,exp+".json")):
        
            # 1 dir == 1 config. experiment == 1 log file analyzed == 1 list of data to extract
            training_info  = load_jsonData(exp+".json", os.path.join(results_path, exp))
            training_settings = get_hyperparams(exp)

            ratio = training_info['val_dsc']/training_info['val_loss']

        else:
            training_settings = get_hyperparams(exp)
            #missing=missing.append(exp.json)
            training_info = {"total_epochs": -1, "steps_x_epoch": -1,"total_job_ranks":-1,
                             "total_training_time":-1,"early_stop_iteration":-1,
                             "val_loss":-1,"val_acc":-1,"val_dsc":-1}
            ratio = -1
            
        output[str(n)] = [training_info, training_settings, ratio]

    # Step_2: Write the big summary with all the experiments in a .csv file
    saveToSummary(outputfile_name,output,outputfile_path)




if __name__ == "__main__":
   main()
