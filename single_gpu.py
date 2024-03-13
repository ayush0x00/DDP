import torch
from torch.utils.data import Dataset, DataLoader
from RandomDataset import RandomData


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: str,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, source, target):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU {self.gpu_id}] Epoch {epoch} | Batch size: {b_sz} | Steps: {len(self.train_data)}")
        for source, target in self.train_data:
            source = source.to(self.gpu_id)
            target = target.to(self.gpu_id)
            self._run_batch(source, target)

    def _save_checkpoint(self, epoch):
        ckp = {}
        ckp["MODEL_STATE"] = self.model.state_dict()
        ckp["OPTIM_STATE"] = self.optimizer.state_dict()
        ckp["EPOCH"] = epoch
        torch.save(ckp, "checkpoint.pt")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch=epoch)

def load_objs():
    dataset = RandomData(784, 60000)
    model = torch.nn.Linear(784, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return dataset, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main(device,total_epochs,save_every):
    dataset,model,optimizer = load_objs()
    data_loader = prepare_dataloader(dataset,64)
    trainer = Trainer(model,data_loader,optimizer,device,save_every)
    trainer.train(total_epochs)


if __name__=='__main__':
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    device = 'cpu'
    main(device,total_epochs,save_every)
        
