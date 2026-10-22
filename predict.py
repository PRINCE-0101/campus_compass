"""Usage: python predict.py 'When does the campus shuttle leave?'"""
import argparse
import json
from core import Predictor

def main():
    parser = argparse.ArgumentParser(description='Campus query intent classifier')
    parser.add_argument('query', nargs='?', help='Query; omit to enter interactive mode')
    args = parser.parse_args()
    try:
        predictor = Predictor()
        if args.query is not None:
            print(json.dumps(predictor.predict(args.query), indent=2))
        else:
            print('Campus Compass — type quit to exit.')
            while True:
                try:
                    text = input('Query > ')
                except (EOFError, KeyboardInterrupt):
                    break
                if text.lower().strip() in {'quit', 'exit'}:
                    break
                try:
                    print(json.dumps(predictor.predict(text), indent=2))
                except ValueError as exc:
                    print(f'Input error: {exc}')
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

if __name__ == '__main__':
    main()
