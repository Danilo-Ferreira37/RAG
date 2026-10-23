from src.models import RagDataset
from src.load import parse_questions
from pydantic import ValidationError
import json
from src.cli import Rag
import fire


def main() -> None:
    try:
        file = "datasets_public/public/AnsweredQuestions/dataset_code_public.json"
        config = parse_questions(file, True)
        fire.Fire(Rag)

    except json.decoder.JSONDecodeError as e:
        print(f"Json File invalid: {e}")
        exit(1)
    except ValidationError as e:
        print(f"Error in input file: {file}")
        print(e)
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    except BaseException as e:
        print(e)
        exit(1)

if __name__ == "__main__":
    main()