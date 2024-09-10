# Deep Learning

## Prerequisits
### Install psbody.mesh Compute Canada

```console
Installing psbody-mesh on compute canada server :

After creating and activating a fresh python virtual environment, install the following,

pip install --upgrade setuptools wheel
pip install numpy==1.26.4+computecanada


1. Clone mesh repository
$ git clone https://github.com/MPI-IS/mesh.git

2. Activate boost on compute canada cluster
$ module load boost

3. Specify boost include dir
$ module show boost

4. Replace content of  requirements.txt, under mesh folder, with :
		setuptools
		numpy
		matplotlib
		scipy
		pyopengl
		pillow
		pyzmq
		pyyaml
		opencv-python==4.10.0

   Replace 7th line of the Makefile with :
		@pip install --no-deps --verbose --no-cache-dir .

   Run : module spider opencv/4.10.0
   Instruction will appear to load a few modules before loading opencv.
   e.g. module load StdEnv/2023  gcc/12.3  openmpi/4.1.5  cuda/12.2
   After following instructions load opencv module :
   module load opencv/4.10.0 

5. Install using Makefile (from inside of mesh folder)
$ BOOST_INCLUDE_DIRS=/path/to/boost/include make all
```

### Install psbody.mesh local

```console
1. Clone mesh repository
$ git clone https://github.com/MPI-IS/mesh.git

2. Change mod
$ chmod 755 /mesh/Makefile

3. Install using Makefile
$ BOOST_INCLUDE_DIRS=/usr/include/boost make all

4. Install OpenGL
$ conda install -c anaconda pyopengl
```
