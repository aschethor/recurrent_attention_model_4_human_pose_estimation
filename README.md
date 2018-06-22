# Recurrent Attention Model for Human Pose Estimation
Master Thesis of: Nils Wandel; Supervised by: Isinsu Katircioglu

Abstract:

In this project we propose a new method for 2D human pose estimation from monocular images using a recurrent attention model. With reference to the paper on "Recurrent Models ofVisual Attention" [Mnih, Heess, Graves, and Kavukcuoglu (2014)], the concept ofmaking predictions relying on an agent that can focus on certain parts of the input image was taken to localize joints of the human body. The agent’s policy was modeled deterministically and extended so that on top ofmoving an attention window in x and y directions, the agent is also able to zoom into the scene. This enables the agent to make precise human pose estimations at varying scales.

The method was tested with an in house boxing dataset and the MPii dataset [Andriluka, Pishchulin, Gehler, and Schiele]. Its performance proved to be comparable to the efficient ConvNet architecture proposed by Tompson [Tompson, Goroshin, Jain, LeCun, and Bregler (2015)] when no additional ground truth information from the MPII dataset is taken into account. In the future such a model could be extended to 3D predictions and predictions over multiple image frames.
