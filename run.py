from adapt.trainer import Trainer

if __name__ == "__main__":

    import wandb

    wandb.init(project="Domain-Generalization")

    domains = ["Art", "Clipart", "Product", "Real World"]
    trainer = Trainer(domains[0], domains[1], True)

    trainer.run(100)
