from config import settings
from trainer import Trainer


def main() -> None:
    trainer = Trainer(settings)
    trainer.fit_and_log()


if __name__ == "__main__":
    main()
