model_name=$1
model_worker_port=$2
num_gpus=$3
if [ -z "$model_name" ]; then
    echo "Please provide the model name as the first argument"
    exit 1
fi
if [ -z "$model_worker_port" ]; then
    echo "Please provide the model worker port as the second argument"
    exit 1
fi
if [ -z "$num_gpus" ]; then
    num_gpus=1
    echo "Number of GPUs is not provided, default to 1"
fi

# hard coded for now
controller_addr=http://34.19.37.54:8888 # hard coded for now, wildvision controller address
# controller_addr=http://127.0.0.1:21002 # for local testing
bore_server_ip=34.19.37.54 # hard coded for now
BORE_LOG_FOLDER="./bore_logs"
mkdir -p $BORE_LOG_FOLDER
# replace / with _ in model name
_model_name=$(echo $model_name | sed 's/\//_/g')
bore_log_file="${BORE_LOG_FOLDER}/bore_output_${_model_name}_${model_worker_port}.log"

if command -v bore &> /dev/null
then
    echo "bore is installed, try assigning a public address for the worker"
else
    echo "bore is not installed, please install it by running the following commands:"
    echo "Step 1: if you did not install rust and cargo, please first run the following command (ignore if you have already installed):"
    echo "curl https://sh.rustup.rs -sSf | sh"
    echo "Step 2: Install bore by running the following command:"
    echo "cargo install --git https://github.com/jdf-prog/bore"
    exit 1
fi
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
python -m lmm_engines.huggingface.model_worker --model-path $model_name --controller ${controller_addr} --port $model_worker_port --worker http://${bore_public_addr} --host=0.0.0.0 --num-gpus $num_gpus
