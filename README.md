# Adversarial Skill Networks: Unsupervised Robot Skill Learning from Video
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/mees/Adversarial-Skill-Networks.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/mees/Adversarial-Skill-Networks/context:python)

This repository is a PyTorch implementation of Adversarial Skill Networks (ASN), an approach for unsupervised skill learning from video. Concretely, our approach learns a task-agnostic skill embedding space from unlabeled multiview videos. We combine a metric learning loss, which utilizes temporal video coherence to learn a state representation, with an entropy regularized adversarial skill-transfer loss. The learned embedding enables training of continuous control policies to solve novel tasks that require the interpolation of previously seen skills. More information at our [project page](http://robotskills.cs.uni-freiburg.de/).

<p align="center">
  <img src="http://robotskills.cs.uni-freiburg.de/images/motivation.png" width="75%"/>
</p>

## Reference
If you find the code helpful please consider citing our work
```
@INPROCEEDINGS{mees20icra_asn,
  author = {Oier Mees and Markus Merklinger and Gabriel Kalweit and Wolfram Burgard},
  title = {Adversarial Skill Networks: Unsupervised Robot Skill Learning from Videos},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year = 2020,
  address = {Paris, France}
}
```

## Installation
We provide a simple package to install ASN preferably in a virtual environment. We have tested the code with Ubuntu 18.04 and PyTorch 1.4.0.
If not installed, install virtualenv and virtualenvwrapper with --user option: <br>
```pip install --user virtualenv virtualenvwrapper```

Add the following virtualenvwrapper settings to your .bashrc:
<pre>
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/.venvproject
export VIRTUALENVWRAPPER_VIRTUALENV_ARGS='--no-site-packages'
source $HOME/.local/bin/virtualenvwrapper.sh
</pre>


Create a virtual environment with Python3: <br>
```mkvirtualenv -p /usr/bin/python3 asn```<br>

Now clone our repo:<br>
```git clone git@github.com:mees/Adversarial-Skill-Networks.git```<br>

After cd-ing to the repo install it with:<br>
```pip install -e .```<br>



## Training
First download and extract the [dataset](http://robotskills.cs.uni-freiburg.de/dataset/real_block_tasks.zip) containing the block tasks into your /tmp/ directory. To start the training you need to specify the location of the dataset and which tasks you want to held out. For example, to train the skill embedding on the tasks 2 block stacking, color pushing and  separate to stack and evaluate it via video alignment on the unseen color stacking task:<br>

```
python train_asn.py --train-dir /tmp/real_combi_task3/videos/train/ --val-dir-metric /tmp/real_combi_task3/videos/val/  --train-filter-tasks cstack
```

## Evaluation
Evaluate the trained ASN model on the video alignment of a novel task and visualize the corresponding t-SNE plots of the learned embedding:<br>

```
python eval_asn.py --load-model pretrained_model/model_best.pth.tar  --val-dir-metric /tmp/real_combi_task3/videos/val/ --task cstack
```

## Pretrained Model
We provide the weights, a log file and a t-SNE visualization for a pretrained model for the default setting [here](asn/pretrained_model.zip). This model achieves an alignment loss of 0.1638 on the unseen color stacking task, which is very close to the 0.165 reported on the paper.



## Dataset
Our  Block Task dataset contains multi-view rgb data of several block manipulation tasks and can be found [here](http://robotskills.cs.uni-freiburg.de/#dataset).

<p>
  <img src="http://robotskills.cs.uni-freiburg.de/images/2block_stack7_view2.gif" width="24.5%" /> <img src="http://robotskills.cs.uni-freiburg.de/images/sort4_view2.gif" width="24.5%"/> <img src="http://robotskills.cs.uni-freiburg.de/images/cstack76_view2.gif" width="24.5%"/>  <img src="http://robotskills.cs.uni-freiburg.de/images/sep64_view2.gif" width="24.5%"/>
</p>

## Recording your own multi-view dataset
We provide a python script to record and synchronize several webcams. First check if your webcams are recognized:<br>

```
ls -ltrh /dev/video*
```

Now start the recording for n webcams:
```
python utils/webcam_dataset_creater.py --ports 0,1 --tag test --display
```
Hit Ctrl-C when done collecting, upon which the script will compile videos for each view
.
## License
For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. For any commercial purpose, please contact the authors. For Pytorch see its respective license.
