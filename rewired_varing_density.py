from my_function import surrogate_model
import multiprocessing as mp

def main():
    total_files = 100
    b=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
    M=2
    k=0
    for batch_start in range(0, total_files, 4):
        processes = []
        batch_end = min(batch_start + 4, total_files)
        for n in range(batch_start, batch_end):
            p = mp.Process(target=surrogate_model, args=(M,n,b,k))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
if __name__ == "__main__":
    main()
