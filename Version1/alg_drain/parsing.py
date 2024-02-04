#!/usr/bin/env python

import sys
from Drain import LogParser
from utils import evaluator
input_dir  = 'data_to_parse/OpenStack_logs/'  # The input directory of log file
output_dir = 'alg_drain/drain_results/OpenStack_results/'  # The output directory of parsing results
log_file   = 'OpenStack_2k.log'  # The input log file name
log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'  # OS log format
# Regular expression list for optional preprocessing (default: [])
regex= [
    #r"[0-9a-fA-F]{8}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{12}",
    r"((\d+\.){3}\d+,?)+",
    r"/.+?\s", 
    r"\d+"
    
]
st         = 0.5  # Similarity threshold
depth      = 5  # Depth of all leaf nodes

parser = LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)

F1_measure, accuracy = evaluator.evaluate(
        groundtruth=output_dir+log_file+"_good_structured.csv",
        parsedresult=output_dir+log_file+"_structured.csv",
    )