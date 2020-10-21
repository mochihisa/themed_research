from pprint import pprint


def text2list(filename):
    res = [line.rstrip('\n') for line in open(filename)]
    res = [s.split('\t') for s in res]
    res.pop(0)
    pprint(res)


def main():
    text2list('spectrum.txt')


if __name__ == "__main__":
    main()
