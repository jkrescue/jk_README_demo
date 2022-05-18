[//]: <> (title: 2D unsteady cylinder flow with continuous time use case tutorial, author: Hui Xiang, Yanbo Zhang, xianghui01@baidu.com)

# 2D unsteady cylinder flow

This guide introduces how to build a PINN model with continuous time method for simulating 2d unsteady flow passing over a cylinder in PaddleScience.

## Use case introduction

The example of 2d unsteady flow over a cylinder simulate solution (pressure and velocity) of following equations.

<div align="center">    
<img src="image/NS.png" width = "400" align=center />
</div>


The following graphs show the velocity in x direction from training the model on given node from openFoam.
<div align="center">    
<img src="image/cylinder_openfoam.png" width = "400" align=center />
</div>

The result simulated from PINNs is shown below, which is similar with that from openfoam.
<div align="center">    
<img src="image/result.png" width = "400" align=center />
</div>

## How to run the model 

The model contains 4 main parts which are datasets, pinn_solver, train or predict code, and dataload module. Before start simulation, the enviornment should be checked.

### Run model

**a. Install PaddlePaddle**

The PaddlePaddle development version is required, and user can choose the appropriate version based on simulating platform(such as in linux os and cuda10.1 platform, `python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html` can be used for installing) on the [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html) 

**a. Download PaddleScience code**
 
        
   - Confirm working directory
        
   - Downlaod paddlescience code from  [github](https://github.com/PaddlePaddle/PaddleScience), git clone also works by the following code:   
        
    git clone https://github.com/PaddlePaddle/PaddleScience.git

**b. Install dependent libraries**
 
    
   - Rename the folder name as PaddleScience if not
    
   - Change working directory to PaddleScience`cd PaddleScience`
    
   - Install dependent libraries by `pip install -r requirements` 
    
**c.Set environment variables**
 
    
   **-Set environment** 
    
   Setting environment by `%env PYTHONPATH=/user_path*/PaddleScience`, and if editing bash files, using `export PYTHONPATH=$PYTHONPATH:/user_path*/PaddleScience/` instead
 
**d. Start simulation**
 
   Before starting simulaiton, the working directory must be confirmed, and execute`cd ./examples/cylinder/2d_unsteady_continuous/`after changing directory to PaddleSciece.  
   **-Loading data**
   
   The trainning data and supervised data are obtained from openFoam that need to be loaded before simulation. User can change the data file in datasets.
   The dataset is stored on a remote server, so user should download datasets firstly.

    !python download_dataset.py
   
   After loading datasets, user can use these data for training the model.

    # Loading data from openfoam 
    path = './datasets/'
    dataloader = cfd.DataLoader(
        path=path,
        N_f=9000,
        N_b=1000,
        time_start=1,
        time_end=50,
        time_nsteps=50)
    training_time_list = dataloader.select_discretized_time(num_time=30)
    
   **-Define fluid properties**
   
   
   The flow domain is set before simulation. The grid was loaded from openFoam.
    <div align="center">    
    <img src="image/cylinder_grid.png" width = "400" align=center />
    </div>
    
   The fluid propery is represented by fluid viscosity nu. Based on the Reynolds number equation `Re=U*D/nu`, the default inlet velocity is a constant 2, and the Reynolds number can be modified by given various viscosity. 
   In this task, the default Reynolds number is 100, and the cylinder diameter is 1, so we have viscosity equals to 0.02.
   User can change nu to different value if needed.
    
   **-Define Loss weight**
   
    
   Loss function consist of weighted eq_loss, bc_loss, ic_loss, outlet_loss and supervised_data_loss. The weight of each loss can be changed during training process.

    PINN = psolver.PysicsInformedNeuralNetwork(
        layers=6, nu=2e-2, bc_weight=10, eq_weight=1, ic_weight=10, supervised_data_weight=10, 
        outlet_weight=1, training_type='half-supervised', checkpoint_path='./examples/cylinder/2D_unsteady/checkpoint/', 
        net_params=net_params, distributed_env=distributed_env)

   **-Define training parameters**
   
    
   A fully connected neural network is used by default. The network information is defined as 10* 50, all parameters can be changed according to the actual scene.

    def initialize_NN(self, num_ins=3, num_outs=3, num_layers=10, hidden_size=50):
        return psci.network.FCNet(
            num_ins=num_ins,
            num_outs=num_outs,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dtype="float32",
            activation='tanh')


   Adam optimizer is employed in this task. The default training epoch numbers and learning rate are as following, and the learning_rate, num_epoch can be changed.

    adm_opt = paddle.optimizer.Adam(learning_rate=1e-5, parameters=PINN.net.parameters())
    PINN.train(num_epoch=10, optimizer=adm_opt)
    
   **-Training Model**
   
   
   This use case has provided the pre-trained network and saved it in the checkpoint folder. The pre-trained network can be trained continuously when defining ` net_params = './checkpoint/pretrained_net_params'` in cylinder2d_unsteady_train.py. If ` net_params = None`，a new training process is about to begin.
      
    net_params = './checkpoint/pretrained_net_params'
    train(net_params=net_params)
   
   **-Predictint Model**
   
   
   After the training, the optimal network will be generated and saved in checkpoint. Creating 
   Selecting the network and execute`!python cylinder2d_unsteady_train.py`.
   vtk files are generated and saved in vtk folder during predicting. These *vtu* files can be displayed in [Paraview](https://www.paraview.org/).

    if __name__ == "__main__":
        net_params = './checkpoint/pretrained_net_params'
        vtk_filename = './vtk/uvp_t_'
        predict_once_for_all(net_params=net_params, vtk_filename=vtk_filename)

