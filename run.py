from adapt.trainer import Trainer
from adapt.config import parse_arguments

if __name__ == "__main__":

    import wandb

    args = parse_arguments()
    wandb.init(project="Domain-Generalization", name=args.dir_name)

    trainer = Trainer(args)
    trainer.run()
