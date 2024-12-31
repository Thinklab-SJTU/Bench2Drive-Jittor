## Follow these steps to install the environment
- **STEP 1: Create Environment**
    ```
    ## python3.8 should be strictly followed.
    conda create -n b2d_zoo python=3.8
    conda activate b2d_zoo
    ```
- **STEP 2: Install Jittor**
    ```
    sudo apt install libomp-dev

    pip install git+https://github.com/Jittor/jittor.git # make sure use the latest version, after commit da45615
    ```
- **STEP 3: Set Environment Variables**
    ```
    ## Suggested GCC Version 9.4. Otherwise, there would be lots of unknown errors.
    export PATH=YOUR_GCC_PATH/bin:$PATH
    ## Suggested CUDA Version 11.8
    export CUDA_HOME=YOUR_CUDA_PATH/
    ```
- **STEP 4: Install CUDA Support for Jittor**
    ```
    ## If you have a GPU and want to enable CUDA acceleration, install CUDA to the Jittor cache
    python -m jittor_utils.install_cuda
    ```
- **STEP 5: Install ninja and packaging**
    ```
    pip install ninja packaging
    ```
- **STEP 6: Install our repo**
    ```
    pip install -r requirements.txt

    ## If there is any error, consider changing the cuda version of the following package
    pip install spconv-cu113
    pip install cupy-cuda113

    pip install -v -e .
    pip install pillow==9.2.0
    pip install cupy
    ```
- **STEP 7: Prepare pretrained weights.**
    create directory `ckpts`

    ```
    mkdir ckpts 
    ```
    Download `resnet50-19c8e357.pth` form [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/resnet50-19c8e357.pth) or [Baidu Cloud](https://pan.baidu.com/s/1LlSrbYvghnv3lOlX1uLU5g?pwd=1234 ) or from Pytorch official website.
  
    Download `r101_dcn_fcos3d_pretrain.pth` form [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/r101_dcn_fcos3d_pretrain.pth) or [Baidu Cloud](https://pan.baidu.com/s/1o7owaQ5G66xqq2S0TldwXQ?pwd=1234) or from BEVFormer official repo.

- **STEP 8: Install CARLA for closed-loop evaluation.**
    ```
    ## Ignore the line about downloading and extracting CARLA if you have already done so.
    mkdir carla
    cd carla
    wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
    tar -xvf CARLA_0.9.15.tar.gz
    cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
    cd .. && bash ImportAssets.sh
    export CARLA_ROOT=YOUR_CARLA_PATH

    ## Important!!! Otherwise, the python environment can not find CARLA package
    echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/YOUR_CONDA_ENV_NAME/lib/python3.7/site-packages/carla.pth # python3.8 works well even if the egg is compiled for python3.7, please set YOUR_CONDA_PATH and YOUR_CONDA_ENV_NAME correctly
    ```
