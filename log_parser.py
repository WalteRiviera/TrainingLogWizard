#!/usr/bin/env python
# 
# All modification made Walter Riviera
# Author: Walter Riviera
# email: walteriviera@gmail.com
#
import datetime
import os
import sys
import re
import argparse
import math
import csv
import json

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch
import matplotlib.backends.backend_pdf
import numpy as np
from random import sample

from optparse import OptionParser

##################################################################################################
#********************************** LOG-PARSER Aux functions ************************************

def get_jobrank(line=""):
    # parse lines like following this example pattern: "[1,7]<stdout>:Epoch 1/50"
    # return job rank as int, i.e. 7
    pattern = re.compile("^\[\d+\,\d+\]")
    findings = pattern.search(line) 
    if (findings):
        filtered_line = findings.group(0)
        numbers = re.findall(r'\d+',filtered_line)   #numbers = [int(s) for s in re.findall(r'\d+',filtered_line)]
        job, rank = numbers[0], numbers[1]
    else:
        print("ERROR: get_jobrank. Aborting")
        sys.exit()
    
    return int(rank)


def get_epoch(line="", mode="epoch"):
    # parse lines like following this example pattern: "[1,7]<stdout>:Epoch 1/50"
    # return epoch if mode == "epoch" or total number of epoch in case mode = "total"
    out = -1
    pattern = re.compile("[0-9]+\\/[0-9]+")
    findings = pattern.search(line) 
    if (findings):
        filtered_line = findings.group(0)
        numbers = filtered_line.split("/")   #numbers = [int(s) for s in re.findall(r'\d+',filtered_line)]
        epoch, total_epochs = numbers[0], numbers[1]
    
    if (mode == "epoch"):
        out = epoch
    elif (mode == "total"):
        out = total_epochs
    else:
        print("ERROR: get_epoch Mode:", mode, " not recognized")
    return int(out)



def get_iteration_step(line="", mode="step"):
    # parse lines like following this example pattern: 
    # return epoch if mode == "epoch" or total number of epoch in case mode = "total"
    out = -1
    pattern = re.compile("\d+\\/\d+ \[")
    findings = pattern.search(line) 
    if (findings):
        indexes = list(findings.span())
        numbers = (line[indexes[0]:(indexes[1]-2)]).split("/")   #numbers = [int(s) for s in re.findall(r'\d+',filtered_line)]
        step = numbers[0]
        total_steps = numbers[1]

        if (mode == "step"):
            out = step
        elif (mode == "total"):
            out = total_steps
    else:
        print("ERROR: get_iteration_step Mode:", mode, " not recognized")
        
    return int(out)


def get_eta(line=""):
    # Parse the string that follows the pattern " ETA: 2:42 "
    # return the total ETA time in seconds
    out = -1
    pattern = re.compile("\d+\:\d+")
    findings = pattern.search(line) 
    if (findings):
        indexes=list(findings.span())
        time = (line[indexes[0]:indexes[1]]).split(":")
        minutes = time[0]
        seconds = time[1]
        out = (int(minutes)*60)+ int(seconds)
    else:
        only_seconds_case = re.compile("\d+s")
        only_seconds = only_seconds_case.search(line)
        if (only_seconds):
            indexes=list(only_seconds.span())
            time = int(line[indexes[0]:indexes[1]-1])  # -1, ensures "s" is not captured
            out = time
    
        else:
            ## TODO: Implement case with hh:mm:ss
            
            print("ERROR: get_eta. Aborting")
            sys.exit()
        
    return out

def get_loss(line=""):

    # Parse the string that follows the pattern " loss: 0.9999 "
    # return the loss value as double
    out = -1
    pattern = re.compile("\d+\.\d+")
    findings = pattern.search(line) 
    if (findings):
        indexes=list(findings.span())
        time = float(line[indexes[0]:indexes[1]])
        out = time
    else:
        print("ERROR: get_loss. Aborting")
        sys.exit()
        
    return out

def get_accuracy(line=""):
    # Parse the string that follows the pattern " accuracy: 0.9999 "
    # return the accuracy value as double
    out = -1
    pattern = re.compile("\d+\.\d+")
    findings = pattern.search(line) 
    if (findings):
        indexes=list(findings.span())
        time = float(line[indexes[0]:indexes[1]])
        out = time
    else:
        print("ERROR: get_accuracy. Aborting")
        sys.exit()
    
    return out

def get_dsc(line=""):
    # Parse the string that follows the pattern:
    # "[1,2]<stdout>:1/13 [=>............................] - ETA: 2:42 - loss: 0.9998 - accuracy: 0.9533 - dsc: 1.7519e-04"
    # return the dsc value as float
    out = -1
    
    pattern = re.compile("dsc\:")
    findings = pattern.search(line) 
    if (findings):
        indexes=list(findings.span())
        dsc_str_portion=line[indexes[0]:]
        dsc_value = dsc_str_portion.split(": ")[1]
        #dsc_value = re.sub("-","",dsc_value)
        
        out = float(dsc_value)
    else:
        print("ERROR: get_dsc. Aborting")
        sys.exit()
        
        
    return out

def get_time(line=""):
    # it extract time field  (in seconds) from a line that follows the pattern:
    # "[1,0]<stdout>:Total time elapsed for training = 2125.779879808426 seconds". 

    out=-1
  
    pattern = re.compile(' \d+.\d+ ')
    check = pattern.search(line) 
    if (check):
        indexes=list(check.span())
        time=line[indexes[0]:indexes[1]]

        out=float(time)

    else:
        print("ERROR: get_dsc. Aborting")
        sys.exit()
        
    return out



def get_earlyStopIteration(line=""):
    
    # it extract epoch number from a line that follows the pattern:
    # "[1,1]<stdout>:Epoch 00016: early stopping". 

    out=-1
  
    pattern = re.compile('\d+\:')
    check = pattern.search(line) 
    if (check):
        indexes=list(check.span())
        epoch=line[indexes[0]:(indexes[1]-1)]

        out=int(epoch)

    else:
        print("ERROR: get_dsc. Aborting")
        sys.exit()
        
    return out



def get_details(line=""):
    # parse lines like following this example pattern: 
    # "[1,2]<stdout>:1/13 [=>............................] - ETA: 2:42 - loss: 0.9998 - accuracy: 0.9533 - dsc: 1.7519e-04"
    # returns numeric fields: job_rank, ETA, loss, acc, dsc
    fields = line.split("-")
    
    rank = get_jobrank(line)
    step = get_iteration_step(line)
    ETA  = get_eta(fields[1])
    loss = get_loss(fields[2])
    acc = get_accuracy(fields[3])
    dsc = get_dsc(line)
    
    return rank, step, ETA, loss, acc, dsc

def get_lastEpochDetails(line=""):

    # parse lines like following this example pattern: 
    # "13/13 [==============================] - 125s 10s/step - loss: 0.9999 - accuracy: 0.9733 - dsc: 8.1184e-05 - val_loss: 0.9999 - val_accuracy: 0.9704 - val_dsc: 7.7363e-05"
    # returns numeric fields: job_rank, avg_secXstep, loss, acc, dsc, val_loss, val_acc, val_dsc
    
    pattern = re.compile(" \- val_loss\:")   
    check = pattern.search(line) 

    if (check):
        indexes=list(check.span())
        
        # Characteristic of this line is that it follows the pattern of a standard
        # training line but it add the val_loss, val_accuracy, val_dsc at the end
        # so the idea is to split the 2 parts to reuse std functions
        std_training_line = line[0:indexes[0]]
        val_scores_line = line[indexes[0]:len(line)]
    
        # Get standard vals
        rank, step, ETA, loss, acc, dsc = get_details(std_training_line)
        avg_timeXstep = math.ceil(ETA/get_iteration_step(std_training_line,"total"))

        # Parsing the string extension with "val_" values
        fields = val_scores_line.split(" - ")

        val_loss = get_loss(fields[1])
        val_acc = get_loss(fields[2])
        val_dsc = get_dsc(val_scores_line)
        
        return rank, step, avg_timeXstep, loss, acc, dsc, val_loss, val_acc, val_dsc
    

def is_epoch(line=""):
    # return true if line follow the pattern "Epoch n/total". False otherwise
    pattern = re.compile('\<stdout\>\:Epoch \d+\/')
    check = pattern.search(line) 
    if (check):
        return True
    else:
        return False

    
def is_time(line=""):
    # return true if line follow the pattern 
    # "[1,0]<stdout>:Total time elapsed for training = 2125.779879808426 seconds". False otherwise
    pattern = re.compile('Total time elapsed for training \=')
    check = pattern.search(line) 
    if (check):
        return True
    else:
        return False

    
def is_earlyStopping(line=""):
    # return true if line follow the pattern 
    # "[1,1]<stdout>:Epoch 00016: early stopping". False otherwise
    pattern = re.compile('Epoch .*\: early stopping')
    check = pattern.search(line) 
    if (check):
        return True
    else:
        return False

def is_training(line=""):
    # return true if line follow the pattern 
    # "[1,2]<stdout>:1/13 [=>............................] - ETA: 2:42 - loss: 0.9998 - accuracy: 0.9533 - dsc: 1.7519e-04". 
    # False otherwise
    pattern = re.compile(r'.*ETA\:.*loss\: \d+.\d+.*accuracy\: \d+.\d+.* dsc\:')
    check = pattern.search(line) 
    if (check):
        return True
    else:
        return False

def is_lastEpochIteration(line=""):
    # return true if line follow the pattern 
    # "[1,2]<stdout>:13/13 [==============================] - 125s 10s/step - loss: 0.9999 - accuracy: 0.9733 - dsc: 8.1184e-05 - val_loss: 0.9999 - val_accuracy: 0.9704 - val_dsc: 7.7363e-05"
    # False otherwise
    pattern = re.compile(r'.*loss\: \d+.\d+.*accuracy\: \d+.\d+.* dsc\:.*val_loss\: \d+.\d+.*val_accuracy\: \d+.\d+.* val_dsc\:')
    check = pattern.search(line) 
    if (check):
        return True
    else:
        return False

def parse_cleanFile(fname="",filepath=""):
    
    ## TODO: write documentation
    #Var Init
    epoch=-1
    total_epochs=-1
    steps_x_epoch=-1
    total_training_time=-1
    early_stop_iteration= -1
    ranks=0
    results={}  # Result dictionary
    result=[]

    # TODO: Add check on the inputs (add "~CLEAN" if not already there)
    cleanfile_name="~CLEAN_"+fname
    inputfile = os.path.join(filepath,cleanfile_name)
    
    with open(inputfile) as f:

        for line in f:
            line=re.sub("\n","",line)
            if (is_epoch(line)):
                epoch = get_epoch(line)
                jr = get_jobrank(line)

                # Initialize the results dict by inferring the num of job ranks automatically
                if not("jobrank_"+str(jr) in results.keys()):
                    results["jobrank_"+str(jr)]=[]
                    ranks=ranks+1
                if (total_epochs == -1):
                    total_epochs = get_epoch(line,"total")  # This run only once


            elif (is_training(line)):
                if (steps_x_epoch == -1):
                    steps_x_epoch = get_iteration_step(line,"total")  # This run only once

                rank, step, ETA, loss, acc, dsc = get_details(line) 
                result = [rank, epoch, step, steps_x_epoch, ETA, loss, acc, dsc]
                key="jobrank_"+str(rank)
                results[key].append(result)# results["process"+str(rank)]=result


            elif (is_lastEpochIteration(line)):
                rank, step, avg_timeXstep, loss, acc, dsc, val_loss, val_acc, val_dsc = get_lastEpochDetails(line)
                result = [rank, epoch, step, steps_x_epoch, avg_timeXstep, loss, acc, dsc, val_loss, val_acc, val_dsc]       
                key="jobrank_"+str(rank)
                results[key].append(result)# results["process"+str(rank)]=result

            elif (is_time(line) and (total_training_time==-1)):
                total_training_time = get_time(line)

            elif (is_earlyStopping(line) and (early_stop_iteration == -1)):
                early_stop_iteration = get_earlyStopIteration(line)



    training_info = {"total_epochs": total_epochs, "steps_x_epoch":steps_x_epoch,"total_job_ranks":ranks,
                     "total_training_time":total_training_time,"early_stop_iteration":early_stop_iteration,
                     "val_loss":val_loss,"val_acc":val_acc,"val_dsc":val_dsc}
    
    
    return training_info, results

def save_csvData(fname="",training_info={},results={},savepath=""):
    
    outfname=re.sub(".log",".csv",fname)
    outputfile = os.path.join(savepath,outfname)
    
    fieldnames=["job_rank", "epoch", "step", "total_steps_x_epoch", "avg_timeXstep", "loss", "acc", "dsc", "val_loss", "val_acc", "val_dsc"]
    with open(outputfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",",quoting=csv.QUOTE_MINIMAL)

        # Write training informations
        writer.writerow(training_info.keys())
        writer.writerow(training_info.values())

        # Write results Dictionary
        writer.writerow(fieldnames)
        for job_rank in results:
            for data in results[job_rank]:
                writer.writerow(data)
        

def save_jsonData(fname="", training_info={}, results={},savepath=""):
    outfname=re.sub(".log",".json",fname)
    outputfile = os.path.join(savepath,outfname)
    
    training_info["results"]=results
    with open(outputfile, 'w') as f:  # writing JSON object
         json.dump(training_info, f)
         
##################################################################################################
#********************************** LOG-CLEANER Aux functions ************************************
def add_newlines(line=""):
    
    delimiter="\[\d+\,\d+\]\<stdout\>\:"
    pattern=re.compile(delimiter)
    
    # Find delimiter matches within line
    matches=[match.group(0) for match in pattern.finditer(line)]
    
    matches=set(matches)      # Keep unique values 
    for m in matches:
        line=re.sub(re.escape(m),"\n"+m,line)

    return line



def clean_logFile(fname="", input_path="", output_path=""):
    
    training_part_found = False
    newline=re.compile("\n")
    
    # Input file name
    inputfile = os.path.join(input_path,fname)
        
    # Output file name
    outputfile_name = "~CLEAN_"+fname
    outputfile = os.path.join(output_path,outputfile_name)
    
    with open(outputfile,'w+') as fout:
        fout.write("CLEANED_VERSION OF: "+fname+"\n")
        with open(inputfile) as fin:
            newline=re.compile("\n")

            for line in fin:
                line=line.strip()
                check=newline.search(line)
                if (check):
                    line=re.sub(r"\n","",line)

                if (is_epoch(line) and not(training_part_found)):
                    training_part_found = True

                if (training_part_found):
                        # Clean the line
                        line=re.sub("\b+","", line)
                        line=add_newlines(line)
                        fout.write(line)
                        #print(line)

                #else:
                #    print("Discarded line: ", line)

                
##################################################################################################
#********************************** PLOTTING Aux functions ************************************

def get_value2vector(history,key,field=8):
    
    ''' It extract all the values at position "field" from 
    the key element in history dict. Field, could be a list
    It return a list with all the values'''
    
    # Field mapping = rank:0, epoch: 1, step:2, steps_x_epoch:3, avg_timeXstep:4, 
    #                 loss:5, acc:6, dsc:7, val_loss:8, val_acc:9, val_dsc:10
    out=[]
    for v in history[key]:
        if (v[2] == v[3]):
            out.append(v[field])

    return out



def plot_jobrankLoss(results={},jobrank="jobrank_0",field="loss"):
    
    #jr=1
    #jobrank="jobrank_"+str(jr)

    fieldMap = {"rank":0, "epoch": 1, "step":2, "steps_x_epoch":3, "avg_timeXstep":4, 
                     "loss":5, "acc":6, "dsc":7, "val_loss":8, "val_acc":9, "val_dsc":10}
    
    training_val = np.array(get_value2vector(results,jobrank,fieldMap["loss"]))
    valid_val = np.array(get_value2vector(results,jobrank,fieldMap["val_loss"]))

    epochs = range(0,len(valid_val))
    fig=plt.figure()
    plt.plot(epochs, training_val, 'g-.', label='loss')
    plt.plot(epochs, valid_val, 'b', label='val_loss')

    plt.title(jobrank+" "+field)
    plt.xlabel('Epochs')
    plt.ylabel(field)
    plt.grid()
    plt.legend()
    plt.show()
    
    
    return fig



def load_csvData(finput="",filepath=""):
    
    ''' Load results and training_info dictionaries from filename defined as "finput" with ".csv" extension'''
    
    csvfile_name=re.sub(".log",".csv",finput)
    inputfile = os.path.join(filepath,csvfile_name)
    
    with open(inputfile, newline='') as fin:
        reader = csv.reader(fin, delimiter=',',quoting=csv.QUOTE_MINIMAL)
        keys = next(reader) 
        values = next(reader) 
        
        training_info={keys[i]:float(values[i]) for i in range(0,len(keys))}
        results={"jobrank_"+str(j):[] for j in range(0,int(training_info["total_job_ranks"]))}
        fieldnames=next(reader)
            
        for row in reader:
            jr = "jobrank_"+str(row[0])
            results[jr].append(row)
            
    return training_info, results
    
    

def load_jsonData(finput="", filepath=""):
    
    ''' Load results and training_info dictionaries from filename defined as "finput" with extension ".json".'''
    
    jsonfile_name=re.sub(".log",".json",finput)
    inputfile = os.path.join(filepath,jsonfile_name)
    
    with open(inputfile, 'r') as f:
        training_info = json.load(f)
    
    results = training_info["results"]
    training_info.pop("results", None)
            
    return training_info, results



def plot_summary(results={},total_epochs=-1,jobrank="jobrank_0",pltname="summary"):

    ## TODO: Add description + check on inputs
    fieldMap = {"rank":0, "epoch": 1, "step":2, "steps_x_epoch":3, "avg_timeXstep":4, 
                     "loss":5, "acc":6, "dsc":7, "val_loss":8, "val_acc":9, "val_dsc":10}
    
    val_loss = np.array(get_value2vector(results,"jobrank_0",fieldMap["val_loss"]))
    val_acc = np.array(get_value2vector(results,"jobrank_0",fieldMap["val_acc"]))
    val_dsc = np.array(get_value2vector(results,"jobrank_0",fieldMap["val_dsc"]))

    epochs = range(0,total_epochs)
    fig= plt.figure()
    ax = plt.subplot(111)
    
    ax.plot(epochs, val_loss, 'b', label='val_loss')
    ax.plot(epochs, val_acc, 'g', label='val_acc')
    ax.plot(epochs, val_dsc, 'r', label='val_dsc')

    plt.title(pltname)
    plt.xlabel('Epochs')
    plt.ylabel('metric_value')
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
        
    plt.close(fig)
    return fig



def plot_summaryDetails(results={},total_epochs=-1,pltname="summaryDetails"):

    ## TODO: Add description + check on inputs
    fieldMap = {"rank":0, "epoch": 1, "step":2, "steps_x_epoch":3, "avg_timeXstep":4, 
                     "loss":5, "acc":6, "dsc":7, "val_loss":8, "val_acc":9, "val_dsc":10}

    jobrank="jobrank_0"
    # Any jobrank would do the job as the "final value" validated
    # are the same for all the jobs. Since there is always at least 1 job, 
    # jobrank_0 can handle mpirun execution logfiles with only 1 instance.

    val_loss= np.array(get_value2vector(results,jobrank,fieldMap["val_loss"]))
    val_acc = np.array(get_value2vector(results,jobrank,fieldMap["val_acc"]))
    val_dsc = np.array(get_value2vector(results,jobrank,fieldMap["val_dsc"]))

    epochs = range(0,total_epochs)
    fig = plt.figure()
    
    for jr in results:

        loss= np.array(get_value2vector(results,jr,fieldMap["loss"]))
        acc = np.array(get_value2vector(results,jr,fieldMap["acc"]))
        dsc = np.array(get_value2vector(results,jr,fieldMap["dsc"]))

        plt.plot(epochs, loss,'-.',color='deepskyblue')
        plt.plot(epochs, acc, '-.',color='lime')
        plt.plot(epochs, dsc, '-.',color='salmon')

    
    plt.plot(epochs, val_loss,'b', label='val_loss')
    plt.plot(epochs, val_acc, 'g', label='val_acc')
    plt.plot(epochs, val_dsc, 'r', label='val_dsc')

    plt.title(pltname)
    plt.xlabel('Epochs')
    plt.ylabel('metrics_values')
    plt.grid()
    #plt.legend()
    #plt.show()
        
    plt.close(fig)
    return fig


def plot_fieldDetailsAx(results={},total_epochs=-1,field="loss",pltname="fieldDetails"):
    
    ## TODO: Add description + check on inputs
    
    import matplotlib._color_data as mcd
    import matplotlib.patches as mpatch
    from random import sample
    
    #colors = [name for name in mcd.CSS4_COLORS          if "xkcd:" + name in mcd.XKCD_COLORS]
    colors   = [name for name in mcd.XKCD_COLORS]
    color_id = sample(range(0, len(colors)),len(results))

    jobrank="jobrank_0"
    fieldMap = {"rank":0, "epoch": 1, "step":2, "steps_x_epoch":3, "avg_timeXstep":4, 
                     "loss":5, "acc":6, "dsc":7, "val_loss":8, "val_acc":9, "val_dsc":10}

    val_field = "val_"+field 
    
    val_metric = np.array(get_value2vector(results,jobrank,fieldMap[val_field])) 

    epochs = range(0,total_epochs)
    fig= plt.figure()
    ax = plt.subplot(111)
    
    plt.ioff()
    
    for n,jr in enumerate(results):

        metric= np.array(get_value2vector(results,jr,fieldMap[field]))
        color = colors[color_id[n]]
        ax.plot(epochs, metric, '-.',color=colors[n], label=field+" "+jr)

    ax.plot(epochs, val_metric, 'b', label=val_field+" "+jr)

    plt.title(pltname)
    plt.xlabel('Epochs')
    plt.ylabel('metrics_values')
    #plt.legend()

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
        
    plt.close(fig)
    return fig


def save_plots2pdf(fname="",plots=[],savepath="",):
    
    outfname=re.sub(".log",".pdf",fname)
    outputfile = os.path.join(savepath,outfname)
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outputfile)
    for p in plots: ## will open an empty extra figure :(
        pdf.savefig(p)
    pdf.close()
    
    
    
def main():
    
    """Run the log_parser"""

    usage = "Usage: log_parser.py [options]"
    parser = OptionParser(usage=usage)
    
    # ways to define the Logfile Name
    parser.add_option("-i","--input"       , dest="logfile", default="", help="Logfile name")
    parser.add_option("-l","--logfile_name"     , dest="logfile", default="", help="Logfile name")
    
    # ways to define the Input Path
    parser.add_option("-p","--input_path"  , dest="logpath", default="", help="Logfile path")
    parser.add_option("-L","--logfile_path", dest="logpath", default="", help="Logfile path")
    
    # ways to define the Output Path
    parser.add_option("-o","--output_path" , dest="savepath",default="",help="Save path for output files")
    parser.add_option("-s","--savepath"    , dest="savepath",default="",help="Save path for output files")
    
    ## Parameter check
    opts, args = parser.parse_args()

    if len(args):
        parser.error("this script does not take any arguments")

    if not(opts.logfile): 
        parser.error("please specify '-i'/''-l'/'--input'/'--logfile_name'")
        sys.exit()
    #-------------------------------------------------------------------------------------------------
    
    
    # Init I/O variables
    logfile_name=opts.logfile

    logfile_path    =opts.logpath
    outputfiles_path=opts.savepath

    
    # Step_0: Clean logfile
    clean_logFile(logfile_name,logfile_path, outputfiles_path)

    # Step_1: Parse the cleaned_logfile
    training_info, results = parse_cleanFile(logfile_name, outputfiles_path)
    # save json file with all the info extracted from the log
    save_jsonData(logfile_name,training_info,results,outputfiles_path)


    # Step_2: Generate plots of interest
    plots=[]
    plots.append( plot_summary(results,training_info["early_stop_iteration"],"jobrank_0","Summary"))
    plots.append( plot_summaryDetails(results,training_info["early_stop_iteration"],"Summary details") )
    plots.append( plot_fieldDetailsAx(results,training_info["early_stop_iteration"],"loss","loss zoom-in") )
    plots.append( plot_fieldDetailsAx(results,training_info["early_stop_iteration"],"acc","acc zoom-in") )
    plots.append( plot_fieldDetailsAx(results,training_info["early_stop_iteration"],"dsc","dsc zoom-in") )
    # Save plots to pdf
    save_plots2pdf(logfile_name,plots,outputfiles_path,)

    # Step_3: Delete temporary files
    cleanfile_name = "~CLEAN_"+logfile_name
    os.remove(os.path.join(outputfiles_path, cleanfile_name))


if __name__ == "__main__":
   main()
