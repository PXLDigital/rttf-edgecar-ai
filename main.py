import os

from ai_training import TrainingClass
from util import load_config


class MainClass():
    def __init__(self):
        print("[MainClass] start constructor")
        self.cfg = load_config()
        print("[MainClass] configuration loaded")
        self.training_class = TrainingClass()


    def process_folders(self):
        print("[MainClass] Processing folders")
        dirs = self.generate_folder_list('./data')
        print('[MainClass] Found current folders : {}'.format(dirs))
        model_name = self.training_class.start_training(self.cfg, dirs, model_name=None, aug=self.cfg.AUGMENT)
        print("[MainClass] Finished training model {}".format(model_name))

    def generate_folder_list(self, folder_name):
        folders = []
        for r, d, f in os.walk(folder_name):
            for folder in d:
                folders.append(os.path.join(r, folder))
        return folders


if __name__ == '__main__':
    main = MainClass()
    main.process_folders()
