## SESAME: Semantic Editing of Scenes by Adding, Manipulating or Erasing Objects

[Paper](https://arxiv.org/abs/2004.04977) [Video](https://twitter.com/i/status/1249680641597362176)

![SESAME teaser](teaser.png-1.png)

### Datasets
Please use the preprocessing steps of [Hierarchical Image Manipulation](https://github.com/xcyan/neurips18_hierchical_image_manipulation) to prepare the datasets.

Then, use a symbolic link (ln -s) or put them in the datasets folder. 

### Pretrained Models

You can find the pretrained models for the tasks of Image Editing and Layout to Image Generation [here](https://owncloud.csem.ch/owncloud/index.php/s/YD0JyynKNEbgde5)! 

### Training/Testing Scripts
Use the files in the scripts folder to train/test the model.

<pre><code>
@misc{ntavelis2020sesame,
    title={SESAME: Semantic Editing of Scenes by Adding, Manipulating or Erasing Objects},
    author={Evangelos Ntavelis and Andrés Romero and Iason Kastanis and Luc Van Gool and Radu Timofte},
    year={2020},
    eprint={2004.04977},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
</code></pre>

We would like to thank the following repos, their code was essential in the developement of this project:

- https://github.com/NVlabs/SPADE
- https://github.com/xcyan/neurips18_hierchical_image_manipulation  
