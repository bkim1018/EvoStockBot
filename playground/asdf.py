import concurrent.futures
import sys
def b(n, e):
    return n*e

def evaluateList(a, e):
    r = []
    with concurrent.futures.ProcessPoolExecutor() as exe:
        fList = [exe.submit(b, q, e) for q in a]
        for future in concurrent.futures.as_completed(fList):
            r.append(future.result())
    return r



if __name__ == '__main__':
    evaluateList(sys.argv)