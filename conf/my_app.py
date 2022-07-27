import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="/", config_name="config")
def my_app(cfg: DictConfig, experiment: str = "train_default") -> None:
    """."""
    print(cfg.train_default.dataset)
    print(OmegaConf.to_yaml(cfg.train_default))


if __name__ == "__main__":
    my_app()
