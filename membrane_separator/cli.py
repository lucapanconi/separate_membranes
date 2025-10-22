import sys
from .run_analysis import main as run_analysis_main

def main():
    sys.argv[0] = "membrane-separator"
    run_analysis_main()

if __name__ == "__main__":
    main()
