
export HOME=/storage/ostrava1/home/stanksil
if [[ $HOSTNAME == zubat* ]]; then
        export THEANO_FLAGS=base_compiledir=/storage/brno2/home/stanksil/theanoc
fi
cd $HOME

module add python27-modules-gcc
cd DQN-for-autonomous-vehicles

python evolution.py
