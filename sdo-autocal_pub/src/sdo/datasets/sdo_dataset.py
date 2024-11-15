import logging
from os import path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sdo.io import sdo_find, sdo_scale
from sdo.pytorch_utilities import to_tensor
from sdo.ds_utility import minmax_normalization

_logger = logging.getLogger(__name__)

class SDO_Dataset(Dataset):
   
    def __init__(
        self,
        data_basedir,
        data_inventory,
        instr=["AIA", "AIA", "HMI"],
        channels=["0171", "0193", "bz"],
        yr_range=[2010, 2018],
        mnt_step=1,
        day_step=1,
        h_step=6,
        min_step=60,
        resolution=512,
        subsample=1,
        test=False,
        test_ratio=0.3,
        shuffle=False,
        normalization=0,
        scaling=True,
        apodize=False,
        holdout=False,
        mm_files=True,
    ):

        assert day_step > 0 and h_step > 0 and min_step > 0

        self.data_basedir = data_basedir
        self.instr = instr
        self.channels = channels
        self.resolution = resolution
        self.subsample = subsample
        self.shuffle = shuffle
        self.yr_range = yr_range
        self.mnt_step = mnt_step
        self.day_step = day_step
        self.h_step = h_step
        self.min_step = min_step
        self.test = test
        self.test_ratio = test_ratio
        self.normalization = normalization
        self.scaling = scaling
        self.apodize = apodize
        self.holdout = holdout
        self.mm_files = mm_files

        _logger.info("apodize={}".format(self.apodize))

        if path.isfile(data_inventory):
            self.data_inventory = data_inventory
        else:
            _logger.warning(
                "A valid inventory file has NOT been passed"
                "If this is not expected check the path."
            )
            self.data_inventory = False

        self.files, self.timestamps = self.create_list_files()

    def find_months(self):
        """select months for training and test based on test ratio"""
        # November and December are kept as holdout
        if not self.holdout:
            months = np.arange(1, 11, self.mnt_step)
            if self.test:
                n_months = int(len(months) * self.test_ratio)
                months = months[-n_months:]
                _logger.info('Testing on months "%s"' % months)
            else:
                n_months = int(len(months) * (1 - self.test_ratio))
                months = months[:n_months]
                _logger.info('Training on months "%s"' % months)
        else:
            months = [11, 12]
        return months

    def create_list_files(self):
        """
        Find path to files that correspond to the requested timestamps.
        Returns: list of lists of strings (file paths), list of tuples (timestamps).
        """
        _logger.info('Loading SDOML from "%s"' % self.data_basedir)
        _logger.info('Loading SDOML inventory file from "%s"' % self.data_inventory)

        indexes = ["year", "month", "day", "hour", "min"]
        yrs = np.arange(self.yr_range[0], self.yr_range[1] + 1)
        months = self.find_months()
        days = np.arange(1, 32, self.day_step)
        hours = np.arange(0, 24, self.h_step)
        minus = np.arange(0, 60, self.min_step)
        tot_timestamps = np.prod([len(x) for x in [yrs, months, days, hours, minus]])

        if self.data_inventory:
            df = pd.read_pickle(self.data_inventory)

            # Apply filters for channels, years, months, days, hours, and minutes
            cond0 = df["channel"].isin(self.channels)
            cond1 = df["year"].isin(yrs)
            cond2 = df["month"].isin(months)
            cond3 = df["day"].isin(days)
            cond4 = df["hour"].isin(hours)
            cond5 = df["min"].isin(minus)

            # Print counts for each condition
            print(f"Files satisfying cond0 (channel): {df[cond0].shape[0]}")
            print(f"Files satisfying cond1 (year): {df[cond1].shape[0]}")
            print(f"Files satisfying cond2 (month): {df[cond2].shape[0]}")
            print(f"Files satisfying cond3 (day): {df[cond3].shape[0]}")
            print(f"Files satisfying cond4 (hour): {df[cond4].shape[0]}")
            print(f"Files satisfying cond5 (minute): {df[cond5].shape[0]}")

            # Combine conditions
            sel_df = df[cond0 & cond1 & cond2 & cond3 & cond4 & cond5]
            n_sel_timestamps = sel_df.groupby(indexes).head(1).shape[0]
            _logger.info(
                f"Timestamps found in the inventory: {n_sel_timestamps} ({float(n_sel_timestamps) / tot_timestamps:.2f})"
            )

            # Further processing: group by indexes, handle file paths, etc.
            grouped_df = sel_df.groupby(indexes).size()
            grouped_df = grouped_df[grouped_df == len(self.channels)].to_frame()

            sel_df = sel_df.reset_index().drop("index", axis=1)
            sel_df = pd.merge(
                grouped_df, sel_df, how="inner", left_on=indexes, right_on=indexes
            )

            s_files = sel_df.sort_values('channel').groupby(indexes)['file_name'].apply(list)
            files = s_files.values.tolist()
            timestamps = s_files.index.tolist()

            discarded_tm = n_sel_timestamps - len(timestamps)
        else:
            _logger.warning(
                "A valid inventory file has not been passed in, be prepared to wait."
            )
            files = []
            timestamps = []
            n_sel_timestamps = 0
            discarded_tm = 0

        for y in yrs:
            for month in months:
                for d in days:
                    for h in hours:
                        for minu in minus:
                            # if a single channel is missing for the combination
                            # of parameters result is -1
                            result = sdo_find(
                                y,
                                month,
                                d,
                                h,
                                minu,
                                initial_size=self.resolution,
                                basedir=self.data_basedir,
                                instrs=self.instr,
                                channels=self.channels,
                            )
                            n_sel_timestamps += n_sel_timestamps
                            if result != -1:
                                files.append(result)
                                timestamp = (y, month, d, h, minu)
                                timestamps.append(timestamp)
                            else:
                                discarded_tm += 1
        return files, timestamps

    def __len__(self):
        """Returns length of dataset."""
        return len(self.files)

    def __getitem__(self, index):
        size = int(self.resolution / self.subsample)
        n_channels = len(self.channels)

        # the original images are NOT bytescaled
        # we directly convert to 32 because the pytorch tensor will need to be 32
        item = np.zeros(shape=(n_channels, size, size), dtype=np.float32)

        img = np.zeros(shape=(size, size), dtype=np.float32)
        for c in range(n_channels):
            if self.mm_files:  # Load the SDOML files depending on which extension used. mm_file = true will load memory maps.
                temp = np.memmap(
                    self.files[index][c],
                    shape=(self.resolution, self.resolution),
                    mode="r",
                    dtype=np.float32,
                )
            else:
                temp = np.load(self.files[index][c], allow_pickle=True)["x"]

            img[:, :] = temp[::self.subsample, ::self.subsample]

            if self.scaling:
                # divide by roughly the mean of the channel
                img = sdo_scale(img, self.channels[c])

            if self.normalization > 0:
                img = self.normalize_by_img(img, self.normalization)

            item[c, :, :] = img

        if self.apodize:
            # Set off limb pixel values to zero
            x = np.arange((img.shape[0]), dtype=np.float) - img.shape[0] / 2 + 0.5
            y = np.arange((img.shape[1]), dtype=np.float) - img.shape[1] / 2 + 0.5
            xgrid = np.ones(shape=(img.shape[1], 1)) @ x.reshape((1, x.shape[0]))
            ygrid = y.reshape((y.shape[0], 1)) @ np.ones(shape=(1, img.shape[0]))
            dist = np.sqrt(xgrid * xgrid + ygrid * ygrid)

            mask = np.ones(shape=dist.shape, dtype=np.float)
            mask = np.where(dist < 200.0 / self.subsample, mask, 0.0)  # Radius of sun at 1 AU is 200*4.8 arcsec

            for c in range(len(self.channels)):
                item[c, :, :] = item[c, :, :] * mask

        timestamps = self.timestamps[index]
        output = [to_tensor(item), to_tensor(timestamps)]

        return output

    def normalize_by_img(self, img, method):
        """ Normalize image based on selected method. """
        if method == 1:
            img = minmax_normalization(img)
        elif method == 2:
            img = (img - np.mean(img)) / np.std(img)
        elif method == 3:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))

        return img
