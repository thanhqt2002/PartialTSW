# SW
python main.py --method sw --num_epoch 300
python visus_shape.py --method sw --plot_style 0
# Db-TSW
python main.py --method twd --num_epoch 300
python visus_shape.py --method twd --plot_style 0
# PartialTSW
python main.py --method twd --twd_unbalanced --max_mass_generated 1.04 --num_epoch 300
python visus_shape.py --method twd --plot_style 2