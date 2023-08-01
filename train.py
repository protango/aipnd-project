import argparse

parser = argparse.ArgumentParser(
    prog='TrainModel', 
    description='Trains an image classifier model.'
)
parser.add_argument('data_dir')
parser.add_argument('--save_dir', default='.')
parser.add_argument('--arch', default='vgg11')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--hidden_units', type=int, default=512)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

print(args)