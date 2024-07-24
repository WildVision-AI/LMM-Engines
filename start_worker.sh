model_name=bczhou/tiny-llava-v1-hf
model_worker_port=31004
controller_addr=http://34.19.37.54:8888 # hard coded for now
bore_server_ip=34.19.37.54 # hard coded for now

echo "Starting worker for model $model_name"
bore local $model_worker_port --to $bore_server_ip & # then copy the assigned public address to the $bore_public_addr variable
python -m lmm_engines.huggingface.model_worker --model-path $model_name --controller ${controller_addr} --port $model_worker_port --worker ${bore_public_addr} --host=127.0.0.1