import os


def main():
    path_train = "D://Code/DL/data/glassesB2/npz/frame/train/"
    path_val = "D://Code/DL/data/glassesB2/npz/frame/val/"

    npz_train = os.listdir(path_train)
    npz_val = os.listdir(path_val)

    npz_train = [os.path.splitext(i)[0] for i in npz_train]
    npz_val = [os.path.splitext(i)[0] for i in npz_val]

    with open(
        "D://Code/DL/TransUNet/lists/list_B2Frame/train.txt", "w", encoding="utf-8"
    ) as f:
        for i in npz_train:
            f.write(i + "\n")
    with open(
        "D://Code/DL/TransUNet/lists/list_B2Frame/val.txt", "w", encoding="utf-8"
    ) as f:
        for i in npz_val:
            f.write(i + "\n")


if __name__ == "__main__":
    main()
