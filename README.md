# Adversarial Skill Networks: Unsupervised Robot Skill Learning from Video

This code implements Adversarial Skill Networks, an approach for unsupervised skill learning from video. Concretely, our approach learns a task-agnostic skill embedding space from unlabeled multiview videos. We combine a metric learning loss, which utilizes temporal video coherence to learn a state representation, with an entropy regularized adversarial skill-transfer loss. The learned embedding enables training of continuous control policies to solve novel tasks that require the interpolation of previously seen skills. More information at our [project page](http://robotskills.cs.uni-freiburg.de/).

<p align="center">
  <img src="http://robotskills.cs.uni-freiburg.de/images/motivation.png" width="75%"/>
</p>

## Reference
If you find the code helpful please consider citing our work 
```
@INPROCEEDINGS{mees20icra_asn,
  author = {Oier Mees and Markus Merklinger and Gabriel Kalweit and Wolfram Burgard},
  title = {Adversarial Skill Networks: Unsupervised Robot Skill Learning from Videos},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation  (ICRA)},
  year = 2020,
  address = {Paris, France}
}

```

## Installation
We provide a simple package to install ASN preferably in a virtual environment. We have tested the code with Ubuntu 18.04.
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
First download and extract the [dataset](http://robotskills.cs.uni-freiburg.de/dataset/real_block_tasks.zip) containing the block tasks into your /tmp/ directory. To start the training you need to specify the location of the dataset and which tasks you want to held out.
```CUDA_VISIBLE_DEVICES=0 python train_asn.py --train-dir /tmp/real_combi_task3/videos/train/ --val-dir-metric /tmp/real_combi_task3/videos/val/ --save-folder ~/tmp/abd_to_c  --train-filter-tasks cstack ```<br>


## Evaluation



## Dataset
Our  Block Task dataset contains multi-view rgb data of several block manipulation tasks and can be found [here](http://robotskills.cs.uni-freiburg.de/#dataset).

<p>
  <img src="http://robotskills.cs.uni-freiburg.de/images/2block_stack7_view2.gif" width="24.5%" /> <img src="http://robotskills.cs.uni-freiburg.de/images/sort4_view2.gif" width="24.5%"/> <img src="http://robotskills.cs.uni-freiburg.de/images/cstack76_view2.gif" width="24.5%"/>  <img src="http://robotskills.cs.uni-freiburg.de/images/sep64_view2.gif" width="24.5%"/>
</p>

## License
For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. For any commercial purpose, please contact the authors. For Pytorch see its respective license.
