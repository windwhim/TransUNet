import os


def list(path_train, path_val, path_train_save, path_val_save):
    npz_train = os.listdir(path_train)
    npz_val = os.listdir(path_val)

    npz_train = [os.path.splitext(i)[0] for i in npz_train]
    npz_val = [os.path.splitext(i)[0] for i in npz_val]

    with open(path_train_save, "w", encoding="utf-8") as f:
        for i in npz_train:
            f.write(i + "\n")
    with open(path_val_save, "w", encoding="utf-8") as f:
        for i in npz_val:
            f.write(i + "\n")


def main():
    path_train = "D://Code/DL/data/glassesB2/npz/frame/train/"
    path_val = "D://Code/DL/data/glassesB2/npz/frame/val/"

    path_train_save = "D://Code/DL/TransUNet/lists/list_B2Frame_npz/train.txt"
    path_val_save = "D://Code/DL/TransUNet/lists/list_B2Frame_npz/val.txt"

    list(path_train, path_val, path_train_save, path_val_save)


if __name__ == "__main__":
    main()
