This is re-implemented DeepSRGR instantiated on ERAN
========

DeepSRGR was proposed in [TACAS'21](https://arxiv.org/abs/2010.07722).
This repo is the re-implementaion of the spurious region guided refinement method proposed in DeepSRGR and is instantiated on DeepPoly domain in ETH Robustness Analyzer for Neural Networks ([ERAN](https://github.com/eth-sri/eran)). 


Requirements 
------------
GNU C compiler, ELINA, Gurobi,

python3.6 or higher (python3.6 might be preferred), tensorflow, numpy.


Installation
------------
Clone the ARENA repository via git as follows:
```
git clone https://github.com/arena-verifier/DeepSRGR.git
cd DeepSRGR
```

The dependencies can be installed as follows (sudo rights might be required):
```
sudo ./install.sh
source gurobi_setup_path.sh
```

Note that to run the system with Gurobi one needs to obtain an academic license from https://user.gurobi.com/download/licenses/free-academic.

To install the remaining python dependencies (numpy and tensorflow), type:

```
pip3 install -r requirements.txt
```


Usage
-------------
To run the MNIST_6_200 experiment, please use the following script
```
cd tf_verify

bash DeepSRGR_script.sh
```
