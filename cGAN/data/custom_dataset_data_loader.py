import torch.utils.data
from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'unaligned': # only unaligned dataset is currently implemented
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset(opt)
    elif opt.dataset_mode == 'aligned':
        raise NotImplementedError("Only unaligned dataset is supported.")
    elif opt.dataset_mode == 'single':
        raise NotImplementedError("Only unaligned dataset is supported.")
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=opt.random_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
