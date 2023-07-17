import numpy as np

def main():
    X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
    print(X)
    print(Y)

    
if __name__ == "__main__":
    main()