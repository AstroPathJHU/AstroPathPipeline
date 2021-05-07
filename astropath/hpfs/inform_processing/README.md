# 5.10. inForm Processing
## 5.10.1. Description
After the images are corrected by the ```flatw``` module and the ```flatw_path``` is created, inForm algorithm's should be created for each antibody separately according to the protocols laid out in **protocol**. Once algorithms are created, each algorithm should be processed through inForm. This process is onerous for a few reasons. First, because of the number of images involved in whole slide scan processing one algorithm on one slide can take on the order of 1 - 6 hour. Since we use separate algorithms for each antibody, processing a 7 color panel could take over 24 hours for all algorithms in a slide to process. Second, we change the path locations and path names of the slides; typical methods for loading all images in a slide, or all slides in a directory, into an inForm batch analysis do not work. In order to automate this processing, JHU installed a set of inForm licenses onto virtual machines located on a server. Code was then written in the programming language AutoIt to simulate mouse clicks and run an algorithms in the batch analysis method for slides with images in the *AstroPath Pipeline* directory structure. Afterward a queued system was developed to process algorithms at scale using this utility. Each of these steps (creating algorithms, setting up VMs, adding algorithms to the queue, and running the code) is detailed in this section. Because there are so many steps and in order to limit the length of this page, documentation was written into different pages and linked here by a table of contents.

## 5.10.2. Contents
- [5.10.3. inForm Cell Analysis® Multipass Phenotyping](docs "Title")
- [5.10.4. Saving Project for the inForm Cell Analysis® Farm]()
- [5.10.5. Setting up the Virtual Machines for inForm]()
- [5.10.6. Adding Slides to the inForm Queue]()
- [5.10.7. inForm Processing Code]()
