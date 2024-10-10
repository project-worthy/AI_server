import pickle
def check_pickle():
    data = pickle.load(open("calibration.pkl","rb"))
    print(data)


if __name__ == "__main__":
    check_pickle()
    
