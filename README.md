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


## Training


## Evaluation



## Dataset
Our  Block Task dataset contains multi-view rgb data of several block manipulation tasks and can be found [here](http://robotskills.cs.uni-freiburg.de/#dataset).

<p>
  <img src="http://robotskills.cs.uni-freiburg.de/images/2block_stack7_view2.gif" width="24%" /> <img src="http://robotskills.cs.uni-freiburg.de/images/sort4_view2.gif" width="24%"/> <img src="http://robotskills.cs.uni-freiburg.de/images/cstack76_view2.gif" width="24%"/>  <img src="http://robotskills.cs.uni-freiburg.de/images/sep64_view2.gif" width="24%"/>
</p>

## License
For academic usage, the code related to the gating layers, our caffe models and utility scripts are released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. For any commercial purpose, please contact the authors. For Pytorch see its respective license.
