#!/bin/bash
# author zijiaozeng@tencent.com
# created: 2019-01-23
cd "$(dirname "$0")"

export PYTHONPATH=$(pwd)/../../
function debug {
if ! [ "x$BASH_DEBUG" == "x" ]
then
echo `date '+% Y% m% d % H:% M:% S'` " [DEBUG] $* "
fi
}
function info() {
echo `date '+% Y% m% d % H:% M:% S'` " [INFO] $* "
}
function error() {
echo `date '+% Y% m% d % H:% M:% S'` " [ERROR] $* "
}
function fatal() {
local msg=$*
error "$msg"
exit 1
}
function usage() {
echo "Usage: $1 -c <conf_file> -s <mode> [-p] <kv_pair> [-d]"
echo "e.g. $1 -c conf/instree.json -s train -p _define.marker_id=0 -d"
}
prog=$0
set -- `getopt -u -o c:s:b:p:d --long help -- "$@"`
is_deamon=false
while true; do
case $1 in
-c) conf_file=$2; shift 2;;
-s) mode=$2; shift 2;;
-b) begin=$2; shift 2;;
-p) kv_pair=$2; shift 2;;
-d) is_deamon=true; shift;;
--help) usage $prog && exit 0; shift; break;;
--) shift; break;;
*) break;;
esac
done
if [ x"$conf_file" == x ] || [ x"$mode" == x ];then
usage && fatal "Input params is invalid!"
fi
info "PYTHONPATH : $PYTHONPATH"
CMD="python main.py -c ${conf_file}"
if [ x"$kv_pair" != x ];then
CMD="$CMD $kv_pair"
fi
if [ x"$begin" != x ];then
pod_index=`expr $begin + ${POD_INDEX}`
CMD="${CMD} _define.marker_id=${pod_index}"
fi
if [ "$mode" == "all" ]
then
CMD=("${CMD} -s train" "${CMD} -s test")
else
CMD=("${CMD} -s $mode")
fi
if $is_deamon; then
CMD="nohup ${CMD} &"
fi
for cmd in ${CMD[@]}
do
info "execute $CMD"
$CMD
done