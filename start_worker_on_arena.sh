model_name=$1
model_worker_port=$2

# hard coded for now
controller_addr=http://34.19.37.54:8888 # hard coded for now, wildvision controller address
# controller_addr=http://127.0.0.1:21002 # for local testing
bore_server_ip=34.19.37.54 # hard coded for now
BORE_LOG_FOLDER="./bore_logs"
mkdir -p $BORE_LOG_FOLDER
bore_log_file="${BORE_LOG_FOLDER}/bore_output_${model_name}_${model_worker_port}.log"
bore local $model_worker_port --to $bore_server_ip > $bore_log_file 2>&1 &
trap "kill $!" EXIT
sleep 5
if bore_public_addr=$(awk -F "listening at " '/listening at/ {print $2}' $bore_log_file | head -n 1); then
    echo "Get assigned a public address for the worker:" $bore_public_addr
else
    echo "Failed to get assigned a public address for the worker, if your worker in running on server with public IP, consider hard coding the public IP by --worker {public_ip}:{port}"
fi

# above code is to get the public address of the worker, hard coded for now


# add CUDA_VISIBLE_DEVICES=0 if you want to specify the GPU
python -m lmm_engines.huggingface.model_worker --model-path $model_name --controller ${controller_addr} --port $model_worker_port --worker http://${bore_public_addr} --host=0.0.0.0
