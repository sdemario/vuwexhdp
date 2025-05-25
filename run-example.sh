#!/bin/bash

python wordcount_mrjob.py -r hadoop hdfs:///user/almeidshel/group/file01 > output.txt
