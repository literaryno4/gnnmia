import argparse


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', choices=[
                        'Cora', 'CiteSeer', 'PubMed'], help='dataset used to train target mode and shadow_model')
    parser.add_argument('--target_model', type=str, default='GCN',
                        choices=['GCN', 'GraphSage', 'GAT'], help='target model of the attack')
    parser.add_argument('--shadow_model', type=str, default='GCN', choices=[
                        'GCN', 'GraphSage', 'GAT'], help='shadow model to imitate target model')
    parser.add_argument('--attack_model', type=str, default='GraphSage', choices=[
                        'GCN', 'GraphSage', 'GAT'], help='attack model to do membership inference')
    parser.add_argument('--data_path', type=str,
                        default='./data', help='data path of dataset')
    parser.add_argument('--logfile_name', type=str,
                        default='./log/log.log', help='log file path')
    parser.add_argument('--epoches', type=int, default=200,
                        help='epoches to train target model and shadow model')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'], help='devic')
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
    parser.add_argument('--wandb', action='store_true',
                        help='Track experiment')

    return parser.parse_args()
