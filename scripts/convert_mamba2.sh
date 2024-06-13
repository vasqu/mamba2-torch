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


echo "Converting mamba2-$1"
echo "Saving results to $2"
echo ""

# lfs is needed for the torch bin file(s)
git lfs install
# pass parameter variant as arg
git clone "https://huggingface.co/state-spaces/mamba2-$1"
echo ""

# converting and removing the old stuff
python convert_mamba2_to_hf.py --input_ssm_dir "mamba2-$1" --output_dir "$2"
rm -rf "mamba2-$1"
echo ""

echo "Finished!"
