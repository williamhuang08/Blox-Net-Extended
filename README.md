In this project, I explore an extension of Blox-Net. [anchor text](https://ieeexplore.ieee.org/document/11127489)

```
@inproceedings{goldberg2025bloxnet,
  title={Blox-Net: Generative Design-for-Robot-Assembly Using VLM Supervision, Physics Simulation, and a Robot with Reset},
  author={Andrew Goldberg and Kavish Kondap and Tianshuang Qiu and Zehan Ma and Letian Fu and Justin Kerr and Huang Huang and Kaiyuan Chen and Kuan Fang and Ken Goldberg},
  booktitle={Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025},
  url={https://arxiv.org/abs/2409.17126},
}
```

I was curious to determine how Blox-Net would perform if it were not limited to just cuboids and cylinders. To answer this question, I tested whether extending Blox-Net to accommodate an additional geometric primitive, the pyramid, would impact the recognizability of the generated designs. In robotics and manufacturing, assembly systems are often motivated by CAD designs. These CAD 3D models are most frequently constructed from geometric primitives that consist of cubes, cylinders, spheres, cones, pyramids, and tori. However, Blox-Net is limited to only 2 of the 6 available geometric primitives. 

To explore the effect of the addition of the pyramid, I first simulated the pyramid in PyBullet with a uniform density of 1000 kg/m^3, a lateral coefficient of 0.5, and a spinning coefficient of 0.2. Because of the added complexity of the pyramid's geometry, I adjusted Blox-Net's VLM prompt and perturbation redesign step to reject structures where objects are stacked on the pyramid's apex. More critically, I updated Blox-Net's perturbation redesign to rigorously test the stability of the structure when objects are placed on the pyramid's faces, which can lead to slippage.

Next, I added 3 additional types of pyramids to the available set of blocks (base of 25x25, height of 25; base of 50x50, height of 40; and a base of 100x60, height of 80). Using this new block set, I explored Blox-Net's generated structural designs for both objects that would intuitively benefit from pyramids (ex., a house) [see figure 1 below] and objects that wouldn't (ex., stairs) [see figure 2 below].

From my preliminary findings, it seems that the addition of more geometric primitives to Blox-Net's available block set yields more recognizable designs for objects that would intuitively benefit from pyramids and does not negatively impact objects that don't. I also hypothesize that the added geometric complexity of additional primitives may affect the 99.2% block placement accuracy that Blox-Net currently achieves. A more abstract idea that I had to extend this direction is to formalize the GDfRA system mathematically using group theory. Specifically, we can represent each primitive as an object in a group action and then model the design process as a product of these groups. Overall, I believe that with more time and integration with the robot arm, more rigorous analysis can be performed to test the recognizability and constructability of the new designs.

<img width="511" height="644" alt="BloxNet_intuitive (1)" src="https://github.com/user-attachments/assets/b6ec5238-f077-4d3c-bf8f-768ad2af4f37" />

<img width="398" height="533" alt="BloxNet_nonintuitive (1)" src="https://github.com/user-attachments/assets/f0b24c4f-ccd5-44be-9beb-32c9c715dd4f" />
