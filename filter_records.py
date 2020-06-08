import glob
import os


def generate_image_file_name(nr):
    f = "{}_cam-image_array_.jpg".format(nr)
    return f


def check_if_image_exists(p, folder_name):
    file = os.path.join(folder_name, p)
    return os.path.exists(file)


def delete_record_file(param):
    os.remove(param)


def main():
    ri = lambda fnm: int(os.path.basename(fnm).split('_')[1].split('.')[0])
    directories = [x[0] for x in os.walk("./data/")]
    for d in directories:
        if d == './data/':
            continue
        record_paths = glob.glob(os.path.join(d, 'record_*.json'))
        record_paths.sort(key=ri)
        for record in record_paths:
            print("process : {}".format(record))
            filename = record.split('/')[-1]
            f = filename.split("_")[1].split(".")[0]
            print("filenumber : {}".format(f))
            if not check_if_image_exists(generate_image_file_name(f), d):
                delete_record_file(record)


if __name__ == '__main__':
    main()
