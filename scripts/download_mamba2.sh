echo "Checking for validity of requested mamba2 model"

# only execute the script if we are sure that it's a valid model
valid_parameter_models=("130m" "370m" "780m" "1.3b" "2.7b")
is_valid=false
for i in "${valid_parameter_models[@]}"
do
    if [ "$i" == "$1" ] ; then
        echo "Requested model is valid"
        echo ""
        is_valid=true
        break
    fi
done

if [ "$is_valid" = false ] ; then
    echo 'Requested model is invalid, exiting..'
    exit 1
fi


echo "Downloading mamba2-$1"
echo "Saving to $2/mamba2-$1"
echo ""

# lfs is needed for the safetensors
git lfs install
# pass parameter variant as arg
git clone "https://huggingface.co/AntonV/mamba2-$1-av" "$2/mamba2-$1"
echo ""

echo "Finished!"
