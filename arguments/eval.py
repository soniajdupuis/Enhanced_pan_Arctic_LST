import configargparse

parser = configargparse.ArgumentParser()

parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path to evaluate')

parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='Number of dataloader worker processes')
parser.add_argument('--batch-size', type=int, default=5) # to get 5x5 output patches per AOI
parser.add_argument('--size', type=int, default=240) # patch size
parser.add_argument('--stide', type=int, default=180) # controls stride and thus overlap of different patches
parser.add_argument('--batch-size', type=int, default=5) # to get 5x5 output patches per AOI
parser.add_argument('--in-memory', default=False, action='store_true', help='Hold data in memory during evaluation')
parser.add_argument('--no_params', default=False, action='store_true', help='Hold data in memory during evaluation')
parser.add_argument('--feature-extractor', type=str, default='UNet', help='Feature extractor for edge potentials')

parser.add_argument('--Npre', type=int, default=8000, help='N learned iterations, but without gradients')
parser.add_argument('--Ntrain', type=int, default=1024, help='N learned iterations with gradients')