# Point Cloud Gradient Flow
This is a source code for Noisy Point Cloud Gradient Flow task in the paper.

## Usage
The first step is training. The source code is in the `main.py` file with these following command line arguments:
```bash
optional arguments:
  --method METHOD       name of method used. e.g, sw, twd
  --num_epoch NUM_EP    number of epochs of training
  --L L                 L
  --n_lines N_LINES     Number of lines in each tree
  --lr LR               learning rate
  --twd_unbalanced      flag indicating whether to use Partial TSW or not.
  --max_mass_generated  v(T)
```

After training, the updated data in each step is saved in `results/{method}` folder.

The second step is plotting figures. The source code is in the `visus_shape.py` file with these following command line arguments:
```bash
optional arguments:
  --method METHOD       name of method used. e.g, sw, twd
  --plot_style STYLE    
```
- 0: used when results folder have only one checkpoint. it will show the evolution at step 100, 200 and 300.
- 1: data at last iteration of all checkpoints in the results folder (for debugging)
- 2: data at step 100, 200, 300 of all checkpoints in the results folder (for debugging)

Example command:

See `exp.sh`