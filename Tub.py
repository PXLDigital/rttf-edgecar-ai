import glob
import json
import os
import random
import sys
import time
from builtins import FileNotFoundError

import numpy as np
import pandas as pd


class Tub(object):

    def __init__(self, path, inputs=None, types=None, user_meta=[]):

        self.path = os.path.expanduser(path)
        # print('path_in_tub:', self.path)
        self.meta_path = os.path.join(self.path, 'meta.json')
        self.exclude_path = os.path.join(self.path, "exclude.json")
        self.df = None

        exists = os.path.exists(self.path)

        if exists:
            # load log and meta
            # print("Tub exists: {}".format(self.path))
            try:
                with open(self.meta_path, 'r') as f:
                    self.meta = json.load(f)
            except FileNotFoundError:
                self.meta = {'inputs': [], 'types': []}

            try:
                with open(self.exclude_path, 'r') as f:
                    excl = json.load(f)  # stored as a list
                    self.exclude = set(excl)
            except FileNotFoundError:
                self.exclude = set()

            try:
                self.current_ix = self.get_last_ix() + 1
            except ValueError:
                self.current_ix = 0

            if 'start' in self.meta:
                self.start_time = self.meta['start']
            else:
                self.start_time = time.time()
                self.meta['start'] = self.start_time

        elif not exists and inputs:
            print('Tub does NOT exist. Creating new tub...')
            self.start_time = time.time()
            # create log and save meta
            os.makedirs(self.path)
            self.meta = {'inputs': inputs, 'types': types, 'start': self.start_time}
            for kv in user_meta:
                kvs = kv.split(":")
                if len(kvs) == 2:
                    self.meta[kvs[0]] = kvs[1]
                # else exception? print message?
            with open(self.meta_path, 'w') as f:
                json.dump(self.meta, f)
            self.current_ix = 0
            self.exclude = set()
            print('New tub created at: {}'.format(self.path))
        else:
            msg = "The tub path you provided doesn't exist and you didnt pass any meta info (inputs & types)" + \
                  "to create a new tub. Please check your tub path or provide meta info to create a new tub."

            raise AttributeError(msg)

    def get_last_ix(self):
        index = self.get_index()
        return max(index)

    def update_df(self):
        df = pd.DataFrame([self.get_json_record(i) for i in self.get_index(shuffled=False)])
        self.df = df

    def get_df(self):
        if self.df is None:
            self.update_df()
        return self.df

    def get_index(self, shuffled=True):
        files = next(os.walk(self.path))[2]
        record_files = [f for f in files if f[:6] == 'record']

        def get_file_ix(file_name):
            try:
                name = file_name.split('.')[0]
                num = int(name.split('_')[1])
            except:
                num = 0
            return num

        nums = [get_file_ix(f) for f in record_files]

        if shuffled:
            random.shuffle(nums)
        else:
            nums = sorted(nums)

        return nums

    @property
    def inputs(self):
        return list(self.meta['inputs'])

    @property
    def types(self):
        return list(self.meta['types'])

    def get_input_type(self, key):
        input_types = dict(zip(self.inputs, self.types))
        return input_types.get(key)

    def write_json_record(self, json_data):
        path = self.get_json_record_path(self.current_ix)
        try:
            with open(path, 'w') as fp:
                json.dump(json_data, fp)
                # print('wrote record:', json_data)
        except TypeError:
            print('troubles with record:', json_data)
        except FileNotFoundError:
            raise
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def get_num_records(self):
        import glob
        files = glob.glob(os.path.join(self.path, 'record_*.json'))
        return len(files)

    def make_record_paths_absolute(self, record_dict):
        # make paths absolute
        d = {}
        for k, v in record_dict.items():
            if type(v) == str:  # filename
                if '.' in v:
                    v = os.path.join(self.path, v)
            d[k] = v

        return d

    def check(self, fix=False):
        '''
        Iterate over all records and make sure we can load them.
        Optionally remove records that cause a problem.
        '''
        print('Checking tub:%s.' % self.path)
        print('Found: %d records.' % self.get_num_records())
        problems = False
        for ix in self.get_index(shuffled=False):
            try:
                self.get_record(ix)
            except:
                problems = True
                if fix == False:
                    print('problems with record:', self.path, ix)
                else:
                    print('problems with record, removing:', self.path, ix)
                    self.remove_record(ix)
        if not problems:
            print("No problems found.")

    def remove_record(self, ix):
        '''
        remove data associate with a record
        '''
        record = self.get_json_record_path(ix)
        os.unlink(record)

    def put_record(self, data):
        """
        Save values like images that can't be saved in the csv log and
        return a record with references to the saved values that can
        be saved in a csv.
        """
        json_data = {}
        self.current_ix += 1

        for key, val in data.items():
            typ = self.get_input_type(key)

            if (val is not None) and (typ == 'float'):
                # in case val is a numpy.float32, which json doesn't like
                json_data[key] = float(val)

            elif typ in ['str', 'float', 'int', 'boolean', 'vector']:
                json_data[key] = val

            elif typ is 'image':
                path = self.make_file_path(key)
                val.save(path)
                json_data[key] = path

            elif typ == 'image_array':
                img = Imag
                e.fromarray(np.uint8(val))
                name = self.make_file_name(key, ext='.jpg')
                img.save(os.path.join(self.path, name))
                json_data[key] = name

            else:
                msg = 'Tub does not know what to do with this type {}'.format(typ)
                raise TypeError(msg)

        json_data['milliseconds'] = int((time.time() - self.start_time) * 1000)

        self.write_json_record(json_data)
        return self.current_ix

    def erase_last_n_records(self, num_erase):
        '''
        erase N records from the disc and move current back accordingly
        '''
        last_erase = self.current_ix
        first_erase = last_erase - num_erase
        self.current_ix = first_erase - 1
        if self.current_ix < 0:
            self.current_ix = 0

        for i in range(first_erase, last_erase):
            if i < 0:
                continue
            self.erase_record(i)

    def erase_record(self, i):
        json_path = self.get_json_record_path(i)
        if os.path.exists(json_path):
            os.unlink(json_path)
        img_filename = '%d_cam-image_array_.jpg' % (i)
        img_path = os.path.join(self.path, img_filename)
        if os.path.exists(img_path):
            os.unlink(img_path)

    def get_json_record_path(self, ix):
        return os.path.join(self.path, 'record_' + str(ix) + '.json')

    def get_json_record(self, ix):
        path = self.get_json_record_path(ix)
        try:
            with open(path, 'r') as fp:
                json_data = json.load(fp)
        except UnicodeDecodeError:
            raise Exception('bad record: %d. You may want to run `python manage.py check --fix`' % ix)
        except FileNotFoundError:
            raise
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

        record_dict = self.make_record_paths_absolute(json_data)
        return record_dict

    def get_record(self, ix):

        json_data = self.get_json_record(ix)
        data = self.read_record(json_data)
        return data

    def read_record(self, record_dict):
        data = {}
        for key, val in record_dict.items():
            typ = self.get_input_type(key)

            # load objects that were saved as separate files
            if typ == 'image_array':
                img = Image.open((val))
                val = np.array(img)

            data[key] = val

        return data

    def gather_records(self):
        ri = lambda fnm: int(os.path.basename(fnm).split('_')[1].split('.')[0])

        record_paths = glob.glob(os.path.join(self.path, 'record_*.json'))
        if len(self.exclude) > 0:
            record_paths = [f for f in record_paths if ri(f) not in self.exclude]
        record_paths.sort(key=ri)
        return record_paths

    def make_file_name(self, key, ext='.png'):
        name = '_'.join([str(self.current_ix), key, ext])
        name = name = name.replace('/', '-')
        return name

    def delete(self):
        """ Delete the folder and files for this tub. """
        import shutil
        shutil.rmtree(self.path)

    def shutdown(self):
        pass

    def excluded(self, index):
        return index in self.exclude

    def exclude_index(self, index):
        self.exclude.add(index)

    def include_index(self, index):
        try:
            self.exclude.remove(index)
        except:
            pass

    def write_exclude(self):
        if 0 == len(self.exclude):
            # If the exclude set is empty don't leave an empty file around.
            if os.path.exists(self.exclude_path):
                os.unlink(self.exclude_path)
        else:
            with open(self.exclude_path, 'w') as f:
                json.dump(list(self.exclude), f)

    def get_record_gen(self, record_transform=None, shuffle=True, df=None):

        if df is None:
            df = self.get_df()

        while True:
            for _, row in self.df.iterrows():
                if shuffle:
                    record_dict = df.sample(n=1).to_dict(orient='record')[0]
                else:
                    record_dict = row

                if record_transform:
                    record_dict = record_transform(record_dict)

                record_dict = self.read_record(record_dict)

                yield record_dict

    def get_batch_gen(self, keys, record_transform=None, batch_size=128, shuffle=True, df=None):

        record_gen = self.get_record_gen(record_transform, shuffle=shuffle, df=df)

        if keys == None:
            keys = list(self.df.columns)

        while True:
            record_list = []
            for _ in range(batch_size):
                record_list.append(next(record_gen))

            batch_arrays = {}
            for i, k in enumerate(keys):
                arr = np.array([r[k] for r in record_list])
                # if len(arr.shape) == 1:
                #    arr = arr.reshape(arr.shape + (1,))
                batch_arrays[k] = arr

            yield batch_arrays

    def get_train_gen(self, X_keys, Y_keys, batch_size=128, record_transform=None, df=None):

        batch_gen = self.get_batch_gen(X_keys + Y_keys,
                                       batch_size=batch_size, record_transform=record_transform, df=df)

        while True:
            batch = next(batch_gen)
            X = [batch[k] for k in X_keys]
            Y = [batch[k] for k in Y_keys]
            yield X, Y

    def get_train_val_gen(self, X_keys, Y_keys, batch_size=128, record_transform=None, train_frac=.8):
        train_df = train = self.df.sample(frac=train_frac, random_state=200)
        val_df = self.df.drop(train_df.index)

        train_gen = self.get_train_gen(X_keys=X_keys, Y_keys=Y_keys, batch_size=batch_size,
                                       record_transform=record_transform, df=train_df)

        val_gen = self.get_train_gen(X_keys=X_keys, Y_keys=Y_keys, batch_size=batch_size,
                                     record_transform=record_transform, df=val_df)

        return train_gen, val_gen
