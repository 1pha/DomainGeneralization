import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    model_name_or_path: str = field(
        default="mobilenetv3_rw",
        metadata={"help": "Model name used in timm. Use mobilenet for default"},
    )
    num_epoch: int = field(default=20, metadata={"help": "Number of epochs."})
    src_data: int = field(default=0, metadata={"help": "Which data to use as source."})
    tgt_data: int = field(default=0, metadata={"help": "Which data to use as target."})
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate."})
    dir_name: str = field(
        default=None, metadata={"help": "Name of directory being used for output."}
    )
    embed_dim: int = field(
        default=1280,
        metadata={
            "help": "Embedding dimension being used in the linear classifier. Needs revision if you change the model."
        },
    )
    seed: int = field(default=42, metadata={"help": "Fixate seed."})
    wandb: bool = field(default=True, metadata={"help": "Wheter to use wandb or not."})

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def __repr__(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def load_config(self, json_file, arg_name=None):

        msg = "Load Configuration"
        msg += "" if arg_name is None else f" {arg_name}"
        logger.info(msg)

        try:
            for k, v in json_file.items():
                setattr(self, k, v)
        except:
            logger.warn("Failed to Load Configuration")
            raise


def save_config(output_dir, **kwargs):

    logger.info("Parse Arguments into dict.")
    arguments = dict()
    for arg_name, args in kwargs.items():
        arguments[arg_name] = args.to_dict()

    with open(Path(f"{output_dir}/config.json"), "w") as f:
        json.dump(arguments, f)


def load_config(output_dir):

    json_fname = Path(f"{output_dir}/config.json")
    with open(json_fname, "r") as f:
        return json.load(f)

    # TODO needs revision


def parse_arguments():

    from transformers import HfArgumentParser, set_seed

    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Fixate Seed
    set_seed(args.seed)

    domains = ["Art", "Clipart", "Product", "Real World"]
    args.dir_name = f"{args.model_name_or_path}_src({domains[args.src_data]})_tgt({domains[args.tgt_data]})"

    return args


if __name__ == "__main__":

    args = parse_arguments()
    print(args)
