# custom_trainer
    This class manages training and general utilities for a Pytorch model.

    The motivation for using this class is the following:
        I) Inheritance allows user defined class for complex models and / or training by only adding minimal code
            1) Compartmentalization - the methods can be easily extended and overridden
            2) Flexible containers - the trainer uses dictionaries that can handle complex inputs / outputs
            3) Hooks that grant further possibilities of customization
        II) Already implemented complex functionalities during training:
            1) Scheduler for the learning rate decay and different learning rates for different parameters' groups
            2) Mixed precision training using torch.cuda.amp
            3) Visualization of the learning curves using the visdom library
        III) Utilities for logging, metrics and investigation of the model's outputs
            1) Simplified pipeline for logging, saving and loading a model
            2) The class attempts full model documentation
            3) DictList allows indexing of complex outputs for output exploration
