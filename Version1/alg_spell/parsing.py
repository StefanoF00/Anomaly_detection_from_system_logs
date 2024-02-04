#!/usr/bin/env python


from Spell import LogParser
import argparse
from utils import evaluator
input_dir  = 'data_to_parse/OpenStack_logs/'  # The input directory of log file
output_dir = 'alg_spell/spell_results/OpenStack_2k_results/'  # The output directory of parsing results
log_file   = 'OpenStack_2k.log'  # The input log file name

#log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
log_format = '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>' #HPC log format
#log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>' #BGL log format
#log_format = '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component> <PID> <Content>' #Thunderbird log format
log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'

regex      = [
              #r"[0-9a-fA-F]{8}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{12}",
              r"((\d+\.){3}\d+,?)+", 
              r"/.+?\s", 
              r"\d+"
]  
parser = argparse.ArgumentParser()          # Message type threshold (default: 0.5)
parser.add_argument('-tau', default=0.5, type=float)
args = parser.parse_args()
tau = args.tau
print('tau={}'.format(str(tau)))
parser = LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
parser.parse(log_file)

F1_measure, accuracy = evaluator.evaluate(
        groundtruth=output_dir+log_file+"_good_structured.csv",
        parsedresult=output_dir+log_file+"_structured.csv",
    )


