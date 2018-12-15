Tutorial: How to Configure the Development Environment on Google Cloud Platform
-------------------------------------------------

This tutorial is to teach how to configure a development environment for the GPU training of Keras neural networks with MXNet as backend on Google Cloud Platform.

**Contents**
- [Start an Instance on Google Cloud Platform](#start-an-instance-on-google-cloud-platform)
  * [Create a Google Cloud account](#create-a-google-cloud-account)
  * [Create a new VM instance](#create-a-new-vm-instance)
- [Configure Python-related Environment and File Transfer](#configure-python-related-environment-and-file-transfer)
  * [Install Anaconda](#install-anaconda)
  * [Set IP address to static](#set-ip-address-to-static)
  * [Open Jupyter Notebook in a web browser](#open-jupyter-notebook-in-a-web-browser)
  * [File Transfer via WinSCP](#file-transfer-via-winscp)
- [Keras using MXNet as backend](#keras-using-mxnet-as-backend)
  * [Install CUDA](#install-cuda)
  * [Install cuDNN](#install-cudnn)
  * [Check if everything is fine](#check-if-everything-is-fine)
  * [Install MXNet](#install-mxnet)
  * [Install Keras with MXNet backend](#install-keras-with-mxnet-backend)
  * [Configure Keras backend and image_data_format](#configure-keras-backend-and-image-data-format)
  * [Validate the Installation](#validate-the-installation)
- [Related Links](#related-links)





## Start an Instance on Google Cloud Platform

### Create a Google Cloud account
Enter [https://cloud.google.com/](<https://cloud.google.com/>)
Click "Try free"

![clipboard](https://i.imgur.com/k28OC2o.png)

Fill in billing information, Google Cloud will provide $300 credit for free

![clipboard](https://i.imgur.com/toy4VrL.png)

Then we can enter console

### Create a new VM instance

Click Compute Engine on the left tab

![clipboard](https://i.imgur.com/ltBhytz.png)

Create a new project, Google Cloud will automatically generate a project ID for us

![clipboard](https://i.imgur.com/0vSocIN.png)

Then create a new VM instance

![clipboard](https://i.imgur.com/2e9sKgx.png)

We need change some options here. First, select a proper Region and Zone

 ![clipboard](https://i.imgur.com/8VWnpxi.png)

We can customize the machine type, here we choose 4 vCPU and 1 NVIDA Tesla P100 GPU

![clipboard](https://i.imgur.com/w7UO3zF.png)

We can select boot disk. Since we need install several large package later, here we'd like to have a large disk. Here we choose Ubuntu 16.04 LTS, 30GB

![clipboard](https://i.imgur.com/WqcJdRN.png)

![clipboard](https://i.imgur.com/ZLz0Jjb.png)

Select "Allow HTTP traffic" and "Allow HTTPS traffic" check box in Firewall session, click "Create", wait a few minute for create

![clipboard](https://i.imgur.com/PDIRp0n.png)

Here we got our new VM instance, we can directly click the "SSH" to connect to the instance. Don't forget to stop the instance after finish the task, or Google Cloud will charge you continuously

![clipboard](https://i.imgur.com/WvG3Hpv.png)

![clipboard](https://i.imgur.com/O4AXYLx.png)




## Configure Python-related Environment and File Transfer

### Install Anaconda

>Why do we need Anaconda?
>Anaconda contains python core, jupyter notebook(as IDE), and several useful packages. It is easy for us not only to install but also to manage the development environment.

Go to the Anaconda Downloads page([here](https://www.anaconda.com/download/#linux)), choose the linux version and **copy** the download link address, for now(Oct 2018) the link is `https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh`

Open the SSH of your VM instance, find a path where you want anaconda installed to, use `curl` to download the link we just copied.
```bash
  $ curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
```

Run what the `.sh` file when it's downloaded.
```bash 
  $ bash Anaconda3-5.2.0-Linux-x86_64.sh
```

Follow the instructions to install anaconda.

Once installed, you can activate the installation with the following command:
```bash
  $ source ~/.bashrc
```

### Set IP address to static
To open jupyter notebook from your local web browser, we need to set the external IP address of our Google Cloud instance from the default--dynamic to static. 

To find where to change this setting, follow the step below: 

`MENU->NETWORKING->VPC network->External IP Addresses`

![clipboard](https://i.imgur.com/as2Ui2f.png)

In the list of your addresses, find the one you would like to change and change the type.

![clipboard](https://i.imgur.com/zFSWVh5.png)

Then you are done. The external IP address of your instance now is static.

### Open Jupyter Notebook in a web browser

First we need to create a new firewall rule.

![clipboard](https://i.imgur.com/Qo09WbO.png)

For 'Protocols and ports', choose Specified protocols and ports and set a tcp number you like. I've chosen tcp:5000 as my port number. 

![clipboard](https://i.imgur.com/Aky3297.png)

Now click on the save button.


Then, in the SSH of your VM, check if you have a Jupyter configuration file by typing this commands:
```bash
  $ ls ~/.jupyter/jupyter_notebook_config.py
```
If it doesn’t exist, create one:
```bash
  $ jupyter notebook --generate-config
```

Now open the config file with vi:
```bash
  $ vi jupyter_notebook_config.py 
```

Then we need to add several lines in it, with the `<Port Number>` you set before. For me, it's `5000`.
```
c = get_config()
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = <Port Number>
```

You can add them anywhere inside this file since most of them are comments.
It should look like this:

![clipboard](https://i.imgur.com/QfR9G49.png)


Now you can run jupyter notebook on your server.
```bash
  $ jupyter notebook
```

You will get something like this:

![clipboard](https://i.imgur.com/Lv6yoqt.png)

Copy the token for later login.

Open your web browser and type your `<static external IP address>:<port number>` like this:

![clipboard](https://i.imgur.com/2nVL49p.png)

Then you can use your token to login jupyter notebook！

![clipboard](https://i.imgur.com/2Q6vc6N.png)


### File Transfer via WinSCP
>Why use WinSCP? WinSCP can help you easily upload and download files between your local and the instance. It is used in the next part of our tutorial. More importantly, we will need it to download notebooks and models.

Before starting you should:

- [Have WinSCP installed](https://winscp.net/eng/docs/guide_install)

Now first, we need to use PuTTYgen tool to generate new key.
  >PuTTYgen installs by default with WinSCP. 

Open PuTTYgen, choose `SSH-2 RSA key` under the `Key` tab.

![clipboard](https://i.imgur.com/aQf5JBB.png)

Click `Generate`
>while generating, remember move the mouse over the blank area.

![clipboard](https://i.imgur.com/l5XwhB2.png)


When the generating is done, set the key comment and passphrase(red box below) and you will have a long text string(red box above), copy it.

![clipboard](https://i.imgur.com/MPM5uFo.png)

Then click the save button to save your private key to your local.

![clipboard](https://i.imgur.com/PSIRLor.png)


Go to Google Cloud Platform, click on your instance to enter the detail page.

![clipboard](https://i.imgur.com/4k8N5gR.png)

Then click `edit`, scroll down, under `SSH Keys` click `show and edit`, then click `Add Item` and paste the long text of previous generated key here.

![clipboard](https://i.imgur.com/uUr9aAf.png)

Scroll down to the bottom, `save` it. You will see a pop-up like this:

![clipboard](https://i.imgur.com/1OogZh3.png)


Now go back to your instance, copy the external IP address.

![clipboard](https://i.imgur.com/bk3CXgB.png)


Open WinSCP from your local, from the left panel, choose `New Site`:

![clipboard](https://i.imgur.com/IQoI45l.png)

Paste your external IP and type you Google user name and password, then click `Advanced`:


![clipboard](https://i.imgur.com/3Lt9lA3.png)

Under advanced setting, choose SSH->Authentication->chose your private key file saved in your local before. Then click `ok`

![clipboard](https://i.imgur.com/3B7iUIN.png)

Back to the login window, click `save` to make it easier for you to log in the next time.

![clipboard](https://i.imgur.com/m0G742Q.png)

Now we can `login`

![clipboard](https://i.imgur.com/uokoIFn.png)

It will ask you to enter you passphrase, the one you set when you created the key.

![clipboard](https://i.imgur.com/sBGYquG.png)


By entering correct passphrase, you are able to connect to the server. Now you are able to easily upload or download files!

![clipboard](https://i.imgur.com/WumRJJo.png)


## Keras using MXNet as backend
After previous actions, we've already got a Python environment installed in our server. Now we are going to configure MXNet based Keras using GPU training.

### Install CUDA

To utilize GPU(NVIDIA branded) for training, we have to install two NVIDIA libraries to setup with GPU support. One of them is CUDA, another is cuDNN. 

First let's install CUDA.

The updated version of CUDA is CUDA10, but normally we use previous version like CUDA8 and CUDA9.0+. In our experiments, we used CUDA9.2, so we will use CUDA9.2 as an example.

>You can also install CUDA 9.2 following the NVIDIA's [installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/).



1. **Download CUDA from NVIDIA**

Since we are using an Ubuntu machine, you have to choose the correct CUDA debian package when trying to download from NVIDIA([link](https://developer.nvidia.com/cuda-92-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal)).

Make sure you're choosing this:

![clipboard](https://i.imgur.com/a0RI01L.png)

Then you will see the 'base installer'

![clipboard](https://i.imgur.com/vDSVRkH.png)

For the choices we made, you should get a deb file named as '`cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64.deb`'

Then put the deb file into your sever folder and run the following commands to install it.
>You can upload and download files between local and sever by using [WinSCP](https://winscp.net/eng/download.php)
```bash
  $ sudo dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64.deb
  $ sudo apt-key add /var/cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64/7fa2af80.pub
  $ sudo apt-get update
  $ sudo apt-get install cuda
```
> Note: the `<version>` of `cuda-repo-<version>` should be replaced by the version info corresponding to the name of the downloaded deb file.


Make sure to add the CUDA install path to `LD_LIBRARY_PATH`. Using the following commands:

```bash
  $ export CUDA_HOME=/usr/local/cuda
  $ export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
  $ export LD_LIBRARY_PATH=${CUDA_HOME}/lib64/:$LD_LIBRARY_PATH
```

### Install cuDNN

Now let's install cuDNN. 

First, to download cuDNN, you have to own an NVIDIA developer account. If you don't, you can register it [here](https://login.nvgs.nvidia.com/v1/create-account?mode=login&key=eyJhbGciOiJIUzI1NiJ9.eyJjZCI6IjE0NDEzMDkyMTcwMjg4NDIzMSIsInN0IjoiQXczT3pVaVZNcjZQT1VqbzRXS0tjbDFXSGk3RW1LSU4iLCJydCI6WyJ1c2VyX3Rva2VuIl0sInN0ZCI6bnVsbCwib3QiOiIyMTQxNDc2MTY2MTg4NDA2NDYiLCJucyI6bnVsbCwiY2xtIjp7IkxvY2FsZSI6ImVuLVVTIn0sImNzIjoiTG9nSW4iLCJzZSI6ImY1WWciLCJyZCI6NTM3OTExMDI4LCJ0aCI6IkxpZ2h0Iiwic2kiOiJPQVVUSFYxX3c1VVU3bVk2Yi03cFZCOFdpWlJwNExHVW0zUGRHNF8xMTIzOTU4MCIsImlkIjoiIiwicG8iOm51bGx9._1bJF3uukchuZCb91b56sZiFuws2JeV8iCJlPETbHGo&client_id=144130921702884231&prompt=default&context=initial&theme=Light&locale=en-US).

Once you got a developer account, you can download cuDNN from [here](https://developer.nvidia.com/cudnn).

Choose the correct version of cuDNN for your CUDA. For us, we installed CUDA9.2, so we should install cuDNN 7.1.4 or 7.2.1.

![clipboard](https://i.imgur.com/Zrv1Dq5.png)


Download all 3 '.deb' files: the runtime library, the developer library, and the code samples library for Ubuntu 16.04.

![clipboard](https://i.imgur.com/90mUJyK.png)


In your download folder, install them in the same order:
```bash
  # the runtime library
  $ sudo dpkg -i libcudnn7_7.1.4.18–1+cuda9.2_amd64.deb

  # the developer library
  $ sudo dpkg -i libcudnn7-dev_7.1.4.18–1+cuda9.2_amd64.deb

  # the code samples
  $ sudo dpkg -i libcudnn7-doc_7.1.4.18–1+cuda9.2_amd64.deb
```


Now we can verify the cuDNN installation:

  1. Copy the code samples somewhere you have write access: 
  <br>`cp -r /usr/src/cudnn_samples_v7/ ~`
  2. Go to the MNIST example code: 
  <br>`cd ~/cudnn_samples_v7/mnistCUDNN`
  3. Compile the MNIST example: 
  <br>`make clean && make`
  4. Run the MNIST example:
  <br>` ./mnistCUDNN` 

If your installation is successful, you should see `Test passed!` at the end of the output.

Now we are exporting environment variables LD_LIBRARY_PATH in your .bashrc fileby putting the following line in the end or your `.bashrc` file.


```bash
  $ export LD_LIBRARY_PATH="LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/cuda/extras/CUPTI/lib64"
```

Last, source it:

```bash
  $ source ~/.bashrc
```



### Check if everything is fine
You can verify your CUDA setup with the following commands:

```bash
  $ nvcc --version
  $ nvidia-smi
```

>Note: this command also can be used to monitor the usage of your GPU(s) whiling training.

It will look like this when training with GPU:
![clipboard](https://i.imgur.com/e4JtFTw.png)


### Install MXNet

You have to install the right mxnet corresponding to your installed CUDA version. For example, if you use CUDA9, the mxnet for you is 'mxnet-cu90'. If CUDA8, then use 'mxnet-cu80'. 

Use the following code to install.

```bash
    $ pip install mxnet-cu92
```
>Since our CUDA version is 9.2, we installed mxnet-cu92 correspondingly.


### Install Keras with MXNet backend

We need to install a special version of Keras called 'keras-mxnet', since we are going to use MXNet as backend for Keras. 

Use the following code:

```bash
    $ pip install keras-mxnet --user
```

### Configure Keras backend and image_data_format


In the previous step, we installed the `keras-mxnet`, by default, the following values are set in the keras config file `keras.json`.
```
backend: mxnet
image_data_format: channels_last
```

Accordong to Keras official documentation:
>We strongly recommend changing the image_data_format to channels_first. MXNet is significantly faster on 'channels_first' data. Default is set to 'channels_last' with an objective to be compatible with majority of existing users of Keras. See [performance tips guide](https://github.com/awslabs/keras-apache-mxnet/blob/master/docs/mxnet_backend/performance_guide.md) for more details.

Thus, we are gonna change the default setting. Usually, you can find `keras.json` file in this path: `.keras/keras.json`

You can use vim to open and edit it when in the same folder.

`sudo vi keras.json`

To edit it, press 'i' to enter edit mode. After changing `image_data_format` to `channels_first`, type ':wq' to save and quit.


### Validate the Installation
You can validate the installation by trying to import Keras in Python terminal and verifying that Keras is using mxnet backend.

```
$ python
    >>> import keras as k
        Using mxnet backend>>>
```

Or something like this if you are using jupyter notebook

![clipboard](https://i.imgur.com/FmT7d1c.png)


## Related Links
- [Keras with Apache MXNet Documentation](https://github.com/awslabs/keras-apache-mxnet/tree/master/docs/mxnet_backend)
- [Install CUDA 9.2 and cuDNN 7.1 for PyTorch (GPU) on Ubuntu 16.04](<https://medium.com/@zhanwenchen/install-cuda-9-2-and-cudnn-7-1-for-tensorflow-pytorch-gpu-on-ubuntu-16-04-1822ab4b2421>)
- [How To Install Anaconda on Ubuntu 18.04](<https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart>)
- [Running Jupyter Notebook on Google Cloud Platform in 15 min](<https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52>)
- [Connecting Securely to Google Compute Engine Server with SFTP](https://winscp.net/eng/docs/guide_google_compute_engine)
- [connect to google compute engine from WinSCP and PuttY](https://www.youtube.com/watch?v=JInSwiG9tqQ)