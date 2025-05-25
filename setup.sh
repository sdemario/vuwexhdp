#!/bin/bash
export HADOOP_VERSION=3.3.6
export HADOOP_HOME=/local/Hadoop/hadoop-$HADOOP_VERSION
#export PATH=${PATH}:$HADOOP_HOME/bin
export PATH=${PATH}:$JAVA_HOME:$HADOOP_HOME/bin:$SPARK_HOME/bin
export PATH=$PATH:$HOME/.local/bin
export HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop
export HADOOP_PREFIX=${HADOOP_HOME}
export SPARK_HOME=/local/spark
export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop

need java8

export LD_LIBRARY_PATH=$HADOOP_PREFIX/lib/native:$JAVA_HOME/jre/lib/amd64/server

source classpath.sh

echo  "bla"


#export JAVA_HOME="/usr/pkg/java/sun-8"
#export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native:$JAVA_HOME/jre/lib/amd64/server


pip install mrjob
