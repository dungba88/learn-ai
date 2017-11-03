import scipy.io as sio

def main():
    data = sio.loadmat('skeleton/a1_s1_t1_skeleton.mat')
    skel = data['d_skel']
    print(skel.shape)

if __name__ == '__main__':
    main()
