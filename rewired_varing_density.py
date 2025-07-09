from my_function import surrogate_model
import multiprocessing as mp

def main():
    total_files = 100
    b=[0.6]
    M=0
    for batch_start in range(0, total_files, 7):
        processes = []
        batch_end = min(batch_start + 7, total_files)
        for n in range(batch_start, batch_end):
            p = mp.Process(target=surrogate_model, args=(M,n+60,b))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
if __name__ == "__main__":
    main()
