import sys


def main():
    data = sys.argv[1]
    with open("usedata.txt", "w") as f:
        f.write(data)

main()
