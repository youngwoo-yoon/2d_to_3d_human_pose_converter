import torch
import pickle

from torch.utils.data import Dataset, DataLoader


class PoseDataset(Dataset):

    def __init__(self, pickle_path):
        print("Reading data '{}'...".format(pickle_path))
        self.raw_data = []
        with open(pickle_path, 'rb') as f:
            self.raw_data = pickle.load(f)
        print('done')

        self.pairs = []

        for skel_2d, skel_3d in zip(self.raw_data['2d'], self.raw_data['3d']):
            self.pairs.append([skel_2d, skel_3d])

        self.raw_data = []  # release memory

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx_upper = [0, 3, 4, 5, 9, 10, 11]  # upper-body joints
        pair = self.pairs[idx]

        # [dim x joints] -> (x1,y1,x2,y2,...)
        inputs = pair[0][:, idx_upper].T.reshape(-1)  # upper-body on 2D
        outputs = pair[1][2::3, idx_upper].T.reshape(-1)  # upper-body on 3D, use only z values

        return torch.from_numpy(inputs).float(), torch.from_numpy(outputs).float()


if __name__ == '__main__':
    # test
    pose_dataset = PoseDataset('panoptic_dataset.pickle')
    train_loader = DataLoader(dataset=pose_dataset, batch_size=3, shuffle=True)

    data_iter = iter(train_loader)
    skel_2d, skel_z = next(data_iter)
    print(skel_2d)
    print(skel_z)
