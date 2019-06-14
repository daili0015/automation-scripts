#!/bin/sh
filename=`ls -t |head -n1|awk '{print $0}'`
echo $filename && scp -P 36000 /data/home/luckyzheng/$filename  zcy@9.73.152.204:/dockerdata/luckyzheng
