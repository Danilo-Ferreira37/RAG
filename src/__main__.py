from src.load import parse_questions
from pydantic import ValidationError
import json
from src.cli import Rag
import fire
from src.execptions import RAGError


def main() -> None:
    try:
        fire.Fire(Rag)

    except json.decoder.JSONDecodeError as e:
        print(f"Json File invalid: {e}")
        exit(1)
    except ValidationError as e:
        print(f"Error: {e}")
        exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except RAGError as e:
        print(f"Error: {e}")
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    except BaseException as e:
        print(e)
        exit(1)

if __name__ == "__main__":
    main()