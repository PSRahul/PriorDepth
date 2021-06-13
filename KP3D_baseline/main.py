from options import KP3DOptions
from trainer import Trainer


options = KP3DOptions()
opts = options.parse()

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
